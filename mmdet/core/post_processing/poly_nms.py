import cv2
import torch
import numpy as np
import Polygon as plg


def get_union(pa, pb):
    return pa.area() + pb.area() - get_intersection(pa, pb) + 1e-4


def get_intersection(pa, pb):
    pc = pa & pb
    return pc.area()


def poly_nms(polys, scores, nms_thr, conf_thr, return_ind=False):
    if len(polys) <= 1:
        polys = [(np.array(poly).reshape(-1, 2)).tolist() for poly in polys]
        scores = np.array(scores).tolist()
        return polys, scores

    polys = [plg.Polygon(np.array(poly).reshape(-1, 2)) for poly in polys]
    scores = np.array(scores)

    inds = np.argsort(scores)[::-1]
    nms_inds = inds
    polys = [polys[i] for i in inds]
    scores = scores[inds]
    inds = np.arange(0, len(polys), dtype=np.int64)
    keep = []
    while len(inds) > 0:
        keep.append(inds[0])
        poly_0 = polys[inds[0]]
        tmp_polys = [polys[i] for i in inds[1:]]
        ious = np.asarray(list(map(lambda x: get_intersection(poly_0, x) / get_union(poly_0, x), tmp_polys)))
        inds = inds[1:][ious < nms_thr]
    polys = [polys[i] for i in keep]
    scores = scores[keep]
    nms_inds = nms_inds[keep]

    # filter by scores
    inds = np.where(scores > conf_thr)[0]
    polys = [(np.array(polys[i]).reshape(-1, 2)).tolist() for i in inds]
    scores = scores[inds].tolist()
    nms_inds = nms_inds[inds]
    ret = [polys, scores]
    if return_ind:
        ret.append(nms_inds)
    return ret
