import numpy as np
from pycocotools.coco import COCO
import mmcv
import os.path as osp

from .custom import CustomDataset
from .pipelines import Compose
from .registry import DATASETS
from .langconv import Converter
import Polygon as plg
import cv2


@DATASETS.register_module
class ReCTSDataset(CustomDataset):
    CLASSES = ('char')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 cache_file=None,
                 char_dict_file=None):
        self.ann_root = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.cache_file = data_root + cache_file
        self.char_dict_file = data_root + char_dict_file
        self.dictmap = mmcv.load(data_root + 'dictmap_to_lower.json')

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.ann_root is None or osp.isabs(self.ann_root)):
                self.ann_root = osp.join(self.data_root, self.ann_root)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root, self.proposal_file)

        # load annotations (and proposals)
        self.img_infos, self.char2label, self.label2char = self.load_annotations()
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs(4)
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)
        self.t2s = Converter('zh-hans')

    def load_annotations(self):
        if not osp.isfile(self.cache_file):
            raise NotImplementedError("Error: cache_file does not exist!")
        else:
            img_infos = mmcv.load(self.cache_file)

        if not osp.isfile(self.char_dict_file):
            raise NotImplementedError("Error: char_dict_file does not exist!")
        else:
            data = mmcv.load(self.char_dict_file)
            char2label = data['char2label']
            label2char = data['label2char']
            print('char num: %d' % len(char2label.keys()))

        return img_infos, char2label, label2char

    def get_ann_info(self, idx):
        img_info = self.img_infos[idx]
        ann_path = osp.join(self.ann_root, img_info['annfile'])
        ann_info = mmcv.load(ann_path)

        return self._parse_ann_info(ann_info, img_info)

    def _filter_imgs(self, min_size=32):
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)

        return valid_inds

    def _parse_ann_info(self, ann_info, img_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info['lines']):
            poly = np.array(ann['points']).reshape(-1, 2)

            if plg.Polygon(poly).area() <= 0:
                continue
            x1, y1 = np.min(poly, 0)
            x2, y2 = np.max(poly, 0)
            bbox = [x1, y1, x2, y2]

            if ann['ignore'] == 1:
                gt_bboxes_ignore.append(bbox)
                continue

            gt_bboxes.append(bbox)
            gt_labels.append(1)
            gt_masks_ann.append([ann['points']])

        for i, ann in enumerate(ann_info['chars']):
            if ann['ignore'] == 1:
                continue

            ch = ann['transcription']
            ch = self.t2s.convert(ch)
            if ch in self.dictmap:
                ch = self.dictmap[ch]
            if ch in ['', ' ', '###'] or ch not in self.char2label:
                continue
            label = self.char2label[ch]

            poly = np.array(ann['points']).reshape(-1, 2)

            if plg.Polygon(poly).area() <= 0:
                continue
            x1, y1 = np.min(poly, 0)
            x2, y2 = np.max(poly, 0)
            bbox = [x1, y1, x2, y2]
            gt_bboxes.append(bbox)
            gt_labels.append(label)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        while len(ann_info['masks']) == 0:
            idx = np.random.randint(len(self.img_infos), size=(1,))[0]
            img_info = self.img_infos[idx]
            ann_info = self.get_ann_info(idx)
        results = dict(
            img_info=img_info,
            ann_info=ann_info,
            annpath=osp.join(self.ann_root, img_info['annfile']))
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.pipeline(results)

        return results

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)

        return self.pipeline(results)
