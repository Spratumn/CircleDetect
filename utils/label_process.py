import json
import math
import numpy as np


def get_bbox_from_json(json_file):
    """ get boxes[x1,y1,x2,y2]  with center and w,h from json file"""
    boxes_dict = json.load(open(json_file, encoding='utf-8'))
    boxes = boxes_dict['shapes']
    # [[x1,y1],[x2,y2]], ...
    boxes = [box['points'] for box in boxes]
    # boxes[0][0][0]  # x1
    # boxes[0][0][1]  # y1
    # boxes[0][1][0]  # x2
    # boxes[0][1][1]  # y2
    boxes = [[box[0][0], box[0][1], box[1][0], box[1][1]]
             for box in boxes]
    return boxes


def get_gaussian_radius(h, w, min_iou=0.7):
    assert 0 < min_iou < 1
    return int((1 - math.sqrt(min_iou)) * math.sqrt(h**2 + w**2))


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.round(np.exp(-(x * x + y * y) / (2 * sigma * sigma)), 7)
    return h


def draw_gaussian(heatmap, center, gaussian, radius):
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


def draw_dense_wh(dense_wh, heatmap_max, center, wh_value, gaussian, radius, is_offset=False):
    diameter = 2 * radius + 1
    wh_value = np.array(wh_value, dtype=np.float32).reshape(-1, 1, 1)
    dim = wh_value.shape[0]
    wider_dense_wh = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * wh_value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        wider_dense_wh[0] = wider_dense_wh[0] - delta.reshape(1, -1)
        wider_dense_wh[1] = wider_dense_wh[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap_max.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap_max[y - top:y + bottom, x - left:x + right]
    masked_dense_wh = dense_wh[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = wider_dense_wh[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        idx = (masked_gaussian > masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_dense_wh = (1 - idx) * masked_dense_wh + idx * masked_reg
    dense_wh[:, y - top:y + bottom, x - left:x + right] = masked_dense_wh
    return dense_wh


