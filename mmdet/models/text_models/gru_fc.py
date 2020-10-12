import numpy as np
import Polygon as plg
import torch
import torch.nn as nn
from ..losses import accuracy

class GRUFC(nn.Module):
    def __init__(self, input_dim, output_dim, gru_num=1, with_bn=False, with_bi=False, dropout=0, with_w=False):
        super(GRUFC, self).__init__()
        if with_bn:
            print('with bn')
        if with_bi:
            print('with bi_rnn')
        if dropout > 0:
            print('with dropout %f'%dropout)
        self.with_bn = with_bn
        self.with_bi = with_bi
        self.with_w = with_w
        hidden_dim = 1024
        self.rnn = nn.GRU(input_dim, hidden_dim, gru_num, bidirectional=with_bi, dropout=dropout)
        self.relu = nn.ReLU(inplace=True)
        if with_bi:
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if self.with_w:
            self.fcw = nn.Linear(hidden_dim, 1)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        if with_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        # self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):
        init_list = [self.fc1, self.fc2, self.fc3]
        if self.with_w:
            init_list.append(self.fcw)
        for m in init_list:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feat=False):
        x = x.transpose(0, 1)
        # self.rnn.flatten_parameters()
        x = self.rnn(x)
        if self.with_bi:
            x = x[1][-2:, :, :].transpose(0, 1).contiguous().view(-1, 1024 * 2)
        else:
            x = x[1][-1, :, :].view(-1, 1024)
        if self.with_bn:
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.relu(self.bn2(self.fc2(x)))
        else:
            x = self.relu(self.fc1(x))
            if return_feat:
                feat = x
            x = self.relu(self.fc2(x))
        if self.with_w:
            w = self.fcw(x)
        x = self.fc3(x)

        ret = [x]
        if self.with_w:
            ret.append(w)
        if return_feat:
            ret.append(feat)
        return ret

    def get_target(self,
                   lines,
                   texts,
                   gt_lines,
                   gt_texts,
                   pos_iou=0.5,
                   text_judge=False):
        def get_union(pa, pb):
            pa_area = pa.area()
            pb_area = pb.area()
            return pa_area + pb_area - get_intersection(pa, pb)

        def get_intersection(pa, pb):
            pInt = pa & pb
            if len(pInt) == 0:
                return 0
            else:
                return pInt.area()

        lines = [plg.Polygon(np.array(line).reshape(-1, 2)) for line in lines]
        gt_lines = [plg.Polygon(np.array(gt_line).reshape(-1, 2)) for gt_line in gt_lines]

        def get_label(line, text):
            for gt_line, gt_text in zip(gt_lines, gt_texts):
                if text == gt_text:
                    return 1
                inter = get_intersection(line, gt_line)
                union = get_union(line, gt_line)
                iou = inter / (union + 1e-6)
                if not text_judge:
                    if iou > pos_iou:
                        return 1
                else:
                    if iou > 0.9:
                        return 1
            return 0

        labels = []
        for line, text in zip(lines, texts):
            labels.append(get_label(line, text))
        labels = np.array(labels)

        return labels


    def loss(self,
             cls_scores,
             labels,
             type=''):
        losses = dict()

        suffix = type
        if len(type) > 0:
            suffix = '_' + type

        losses['loss_cls' + suffix] = self.criterion(cls_scores, labels)
        losses['acc' + suffix] = accuracy(cls_scores, labels)
        # print(losses['loss_cls' + suffix].shape)
        # exit()

        return losses
