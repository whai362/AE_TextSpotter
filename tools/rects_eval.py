import mmcv

import argparse
import copy
import json
import os
from collections import defaultdict
from collections import OrderedDict
import os.path as osp
import Polygon as plg

import numpy as np
from PIL import Image
import editdistance


def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups


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


def cat_best_hmean(gt, predictions, thresholds):
    num_gts = len([g for g in gt if g['ignore'] == False])
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([b['bbox'] for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_trans = {k: np.array([b['trans'] for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_ignored = {k: np.array([b['ignore'] for b in boxes])
                        for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    ned = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(predictions):
        pred_polygon = plg.Polygon(np.array(p['bbox']).reshape(-1, 2))
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_ignored = image_gt_ignored[p['name']]
            gt_checked = image_gt_checked[p['name']]
            gt_trans = image_gt_trans[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            ovmax = 0
            jmax = 0
            for j, gt_box in enumerate(gt_boxes):
                gt_polygon = plg.Polygon(np.array(gt_box).reshape(-1, 2))
                union = get_union(pred_polygon, gt_polygon)
                inter = get_intersection(pred_polygon, gt_polygon)
                overlap = inter / (union + 1e-6)
                if overlap > ovmax:
                    ovmax = overlap
                    jmax = j

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    if gt_ignored[jmax]:
                        tp[i, t] = 0.
                        ned[i, t] = 0
                    else:
                        tp[i, t] = 1.
                        ned[i, t] = 1 - editdistance.eval(p['trans'], gt_trans[jmax]) / \
                                    max(len(p['trans']), len(gt_trans[jmax]))
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.
                    ned[i, t] = 0
            else:
                fp[i, t] = 1.

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    ned = np.cumsum(ned, axis=0) / (fp + num_gts + np.finfo(np.float64).eps)

    recalls = tp / float(num_gts)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    fmeasures = 2 * precisions * recalls / (precisions + recalls + 1e-6)
    best_i = np.argmax(fmeasures)
    print('[Best F-Measure] p: {:.2f}, r: {:.2f}, f: {:.2f}, 1-ned: {:.2f}, best_score_th: {:.3f}'.format(
        float(precisions[best_i]) * 100, float(recalls[best_i]) * 100, float(fmeasures[best_i]) * 100,
        float(ned[best_i]) * 100, predictions[best_i]['score']))

    best_i = np.argmax(ned)
    print('[Best 1-NED]     p: {:.2f}, r: {:.2f}, f: {:.2f}, 1-ned: {:.2f}, best_score_th: {:.3f}'.format(
        float(precisions[best_i]) * 100, float(recalls[best_i]) * 100, float(fmeasures[best_i]) * 100,
        float(ned[best_i]) * 100, predictions[best_i]['score']))


def trans_pred_format(pred):
    bdd = []
    img_name = pred['img_name']
    ponits = pred['points']
    scores = pred['scores']
    texts = pred['texts']
    for i in range(len(ponits)):
        bdd_i = {
            'category': 'text',
            'timestamp': 1000,
            'name': img_name,
            'bbox': np.array(ponits[i]).reshape(-1).tolist(),
            'score': scores[i],
            'trans': texts[i]
        }
        bdd.append(bdd_i)
    return bdd


def trans_gt_format(img_name, gt):
    bdd = []
    for i in range(len(gt)):
        bdd_i = {
            'category': 'text',
            'timestamp': 1000,
            'name': img_name,
            'bbox': gt[i]['points'],
            'score': 1,
            'ignore': gt[i]['ignore'],
            'trans': gt[i]['transcription']
        }
        bdd.append(bdd_i)
    return bdd


def main():
    cfg = mmcv.Config.fromfile('local_configs/debug.py')
    gt_root = cfg.data.val.data_root + cfg.data.val.ann_file
    preds = mmcv.load('tmp.json')

    preds_bdd = []
    gts_bdd = []
    for pred in preds:
        img_name = pred['img_name']
        pred_bdd = trans_pred_format(pred)
        preds_bdd.extend(pred_bdd)

        gt_path = gt_root + img_name.replace('.jpg', '.json')
        gt = mmcv.load(gt_path)['lines']
        gt_bdd = trans_gt_format(img_name, gt)
        gts_bdd.extend(gt_bdd)

    cat_gt = group_by_key(gts_bdd, 'category')
    cat_pred = group_by_key(preds_bdd, 'category')
    thresholds = [0.5]
    cat_best_hmean(cat_gt['text'], cat_pred['text'], thresholds)


if __name__ == '__main__':
    main()
