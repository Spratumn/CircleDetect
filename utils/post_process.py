import numpy as np
from utils.image_process import transform_preds
from utils.nms import nms


def get_pred_depth(depth):
    return depth


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def post_process(dets, c, s, h, w, num_classes, score_thresh):
    # dets: [1, N*K, 6]  det:[x1,y1,x2,y2,score,class_id]
    # return top_preds{1: list of [x1,y1,x2,y2,score], 2: list of [x1,y1,x2,y2,score] ... }
    top_preds = {}

    # transform x1,y1
    dets[:, :2] = transform_preds(
        dets[:, 0:2], c[0], s[0], (w, h))
    # transform x2,y2
    dets[:, 2:4] = transform_preds(
        dets[:, 2:4], c[0], s[0], (w, h))

    # do nms on dets before assign class
    keep = nms(dets, 0.5)
    dets = dets[keep]
    # get bbox and score for every class
    classes = dets[:, -1]
    scores = dets[:, 4]
    for j in range(1, num_classes):
        inds = ((classes == j) * (scores > score_thresh))
        top_preds[j] = np.concatenate([
            dets[inds, :4].astype(np.float32),
            dets[inds, 4:5].astype(np.float32)], axis=1).tolist()
    return top_preds

