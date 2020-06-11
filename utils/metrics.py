import numpy as np


def compute_metrics(pre_bboxes, gt_bboxes):
    # x1,y1,x2,y2
    pre_bboxes = np.array(pre_bboxes)
    gt_bboxes = np.array(gt_bboxes)
    pre_num = pre_bboxes.shape[0]
    gt_num = gt_bboxes.shape[0]

    if pre_num == 0:
        return 0, 0

    tp = np.zeros(pre_num)
    for i in range(pre_num):
        pre_box = pre_bboxes[i]
        max_iou = 0
        for box in gt_bboxes:
            iou = computre_iou(pre_box, box)
            if iou > max_iou:
                max_iou = iou
        if max_iou >= 0.5:
            tp[i] = 1
    tp_num = np.count_nonzero(tp)
    precision = tp_num / pre_num
    recall = tp_num / gt_num
    return precision, recall


def computre_iou(bbx1, bbx2):
    # x1,y1,x2,y2
    # intersection over union
    # if there's no intersection
    if bbx1[0] > bbx2[2] \
            or bbx1[2] < bbx2[0] \
            or bbx1[1] > bbx2[3] \
            or bbx1[3] < bbx2[1]:
        return 0

    x_min = max(bbx1[0], bbx2[0])
    x_max = min(bbx1[2], bbx2[2])
    y_min = max(bbx1[1], bbx2[1])
    y_max = min(bbx1[3], bbx2[3])

    # area of intersection rectangle
    inter_area = (x_max - x_min + 1.0) * (y_max - y_min + 1.0)
    union_area = get_area_from_bbx(bbx1)
    # get iou
    iou = inter_area * 1.0 / union_area
    return iou


def get_area_from_bbx(bbx):
    # x1,y1,x2,y2
    area = (bbx[3]-bbx[1]) * (bbx[2]-bbx[0])
    return area
