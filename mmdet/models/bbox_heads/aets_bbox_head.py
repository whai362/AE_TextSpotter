import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import force_fp32, delta2bbox
from mmdet.ops import nms

from ..registry import HEADS
from ..utils import ConvModule
from ..losses import accuracy
from .bbox_head import BBoxHead


@HEADS.register_module
class AETSBBoxHead(BBoxHead):
    """BBoxHead for AE TextSpotter (AETS)
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 cls_in_channels=None,
                 reg_in_channels=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 dropout=0,
                 *args,
                 **kwargs):
        super(AETSBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs + num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch(
            self.num_cls_convs, self.num_cls_fcs,
            self.shared_out_channels if cls_in_channels is None else cls_in_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch(
            self.num_reg_convs, self.num_reg_fcs,
            self.shared_out_channels if reg_in_channels is None else reg_in_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        self.dropout = nn.Dropout(dropout)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))

            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(AETSBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feat=False):
        if type(x) is not tuple:
            # shared part
            if self.num_shared_convs > 0:
                for conv in self.shared_convs:
                    x = conv(x)

            if self.num_shared_fcs > 0:
                if self.with_avg_pool:
                    x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
                for fc in self.shared_fcs:
                    x = self.relu(fc(x))
            # separate branches
            x_cls = x
            x_reg = x
        else:
            x_cls, x_reg = x
        if return_feat:
            feat = x_cls

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(self.dropout(x_cls)) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        ret = [cls_score, bbox_pred]
        if return_feat:
            ret.append(feat)
        return ret

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             type=''):
        losses = dict()
        suffix = type
        if len(type) > 0:
            suffix = '_' + type
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls' + suffix] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc' + suffix] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox' + suffix] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_det_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        scores, labels = torch.max(scores[:, 1:], dim=1)

        if cfg is None:
            return bboxes, labels
        else:
            proposals = torch.cat([bboxes, scores.unsqueeze(-1)], dim=-1)
            det_bboxes, nms_inds = nms(proposals, cfg.nms.iou_thr)
            det_labels = labels[nms_inds]
            valid = det_bboxes[:, -1] > cfg.score_thr
            det_bboxes = det_bboxes[valid]
            det_labels = det_labels[valid]
            nms_inds = nms_inds[valid]

            return det_bboxes, det_labels, nms_inds
