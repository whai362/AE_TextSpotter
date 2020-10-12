import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from ..registry import HEADS
from .aets_anchor_head import AETSAnchorHead


@HEADS.register_module
class AETSRPNHead(AETSAnchorHead):
    # RPNHead for AE TextSpotter (AETS)
    def __init__(self, in_channels, **kwargs):
        super(AETSRPNHead, self).__init__(2, in_channels, **kwargs)
        self.text_num_anchors = len(self.text_anchor_ratios) * len(self.anchor_scales)
        self.char_num_anchors = len(self.char_anchor_ratios) * len(self.anchor_scales)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)

        return rpn_cls_score[:, :self.text_num_anchors * self.cls_out_channels, :, :], \
               rpn_bbox_pred[:, :self.text_num_anchors * 4, :, :], \
               rpn_cls_score[:, self.text_num_anchors * self.cls_out_channels:, :, :], \
               rpn_bbox_pred[:, self.text_num_anchors * 4:, :, :]

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas, cfg, gt_bboxes_ignore=None, type=''):
        losses = super(AETSRPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore,
            type=type)

        suffix = type
        if len(type) > 0:
            suffix = '_' + type

        return {'loss_rpn_cls' + suffix: losses['loss_cls'], 'loss_rpn_bbox' + suffix: losses['loss_bbox']}

    def get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]

        return proposals
