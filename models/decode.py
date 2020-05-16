from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .utils import _gather_feat, _transpose_and_gather_feat


def plot(array2d):
    plt.imshow(array2d)
    plt.show()


def heatmap_nms(heatmap, kernel=3):
    # do nms with heatmap
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def _left_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _right_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _top_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _bottom_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _h_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _left_aggregate(heat) + \
           aggr_weight * _right_aggregate(heat) + heat


def _v_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _top_aggregate(heat) + \
           aggr_weight * _bottom_aggregate(heat) + heat


'''
# Slow for large number of categories
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
'''


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    batch_size, cat, height, width = scores.size()
    # get top K heatmap value and indx (in 1d array of w*h) of every class
    topk_scores, topk_inds = torch.topk(scores.view(batch_size, cat, -1), K)
    # topk_inds = topk_inds % (height * width)
    # print(topk_inds)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()
    # get top K heatmap value and indx (in 1d array of w*h) of all class same as argmax of c channel
    topk_score, topk_ind = torch.topk(topk_scores.view(batch_size, -1), K)
    # belong to which class
    topk_clses = (topk_ind // K)
    topk_inds = _gather_feat(topk_inds.view(batch_size, -1, 1), topk_ind).view(batch_size, K)
    topk_ys = _gather_feat(topk_ys.view(batch_size, -1, 1), topk_ind).view(batch_size, K)
    topk_xs = _gather_feat(topk_xs.view(batch_size, -1, 1), topk_ind).view(batch_size, K)
    # print(topk_score)  # confidences
    # print(topk_clses)  # class_index
    # print(topk_ys, topk_xs)  # center
    # print(topk_inds)  # topk_inds[class_index]

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode(heatmap, wh, offset=None, K=100):
    """

    """
    batch_size, C, H, W = heatmap.size()
    # perform nms on heatmaps by maxpooling2d
    heatmap = heatmap_nms(heatmap)

    scores, inds, clses, ys, xs = _topk(heatmap, K=K)
    # scores:(N,k)
    # inds:(N,k)
    # clses:(N,k)
    # ys:(N,k)
    # xs:(N,k)

    # get predicted wh by inds
    wh = _transpose_and_gather_feat(wh, inds)

    # correct center with offset
    if offset is not None:
        # (N, 2, h, w)-> (N, k, 2)
        offset = _transpose_and_gather_feat(offset, inds)
        xs = xs.view(batch_size, K, 1) + offset[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + offset[:, :, 1:2]

    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)
    # get [x1,y1,x2,y2] from center and wh
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections

