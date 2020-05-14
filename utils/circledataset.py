import pandas as pd
import numpy as np
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
from torch.utils.data import Dataset


def to_float(x):
    return float("{:.2f}".format(x))


def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


class CircleDataset(Dataset):
    """
    load images and labels
    images: add augment
    labels: boxes:[[x1,y1],[x2,y2]]-->[[xc,yc],w,h]-->gaussian
    """
    def __init__(self, cfg, phase):
        super(CircleDataset, self).__init__()
        # read csv as data index
        self.cfg = cfg
        # for circle detection
        # self.data_paths = pd.read_csv(os.path.join(self.cfg.DATA_DIR, '{}.csv'.format(phase)),
        #                               header=None,
        #                               names=["image", "json"])
        # self.image_paths = self.data_paths["image"].values[1:]
        # self.json_paths = self.data_paths["json"].values[1:]

        # for dog detection
        self.data_paths = pd.read_csv(os.path.join(self.cfg.DATA_DIR, '{}.csv'.format(phase)),
                                      header=None,
                                      names=["image", "bbox"])
        self.image_paths = self.data_paths["image"].values[1:]
        self.bbox = self.data_paths["bbox"].values[1:]

        self.max_objs = 128
        self.class_name = ['__background__', 'circle']

        # TODO
        # self._valid_ids = [
        #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
        #     14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        #     24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        #     37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
        #     48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        #     58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
        #     72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        #     82, 84, 85, 86, 87, 88, 89, 90]
        # self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        # self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
        #                   for v in range(1, self.cfg.NUM_CLASSES + 1)]

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.phase = phase
        self.num_samples = len(self.image_paths)
        print('Loaded {} samples'.format(self.num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        # 1.source image process
        # 1.1 load image by index
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        # keep the original resolution when test
        if self.phase == 'test':
            input_h = (height | self.cfg.PAD) + 1
            input_w = (width | self.cfg.PAD) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        # fix resolution when train and eval
        elif (self.phase == 'train') or (self.phase == 'eval'):
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.cfg.INPUT_SIZE
        else:
            raise ValueError('{} is invalid'.format(self.phase))

        flipped = False
        if self.phase == 'train':
            # not use random crop augment
            if not self.cfg.RAND_CROP:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = get_border(128, img.shape[1])
                h_border = get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.cfg.SCALE
                cf = self.cfg.SHIFT
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.cfg.FLIP:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1
        # 1.2 几何变化，改变img尺寸
        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        # 1.3 color augment
        if self.phase == 'train' and self.cfg.COLOR_AUG:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        # 1.4 normalization
        inp = (inp - self.cfg.DATA_MEAN) / self.cfg.DATA_STD
        # [h,w,c]-->[c,h,w]
        inp = inp.transpose(2, 0, 1)

        # 2 create output hm,wh,reg
        # 2.1 output size: input//opt.down_ratio
        output_h = input_h // self.cfg.DOWN_RATIO
        output_w = input_w // self.cfg.DOWN_RATIO
        num_classes = self.cfg.NUM_CLASS

        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        # 2.2 hm,wh,reg
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs,), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs,), dtype=np.uint8)

        # 3.gt label process
        gt_det = []
        # for circle detection
        # json_path = self.json_paths[index]
        # boxes_dict = json.load(open(json_path, encoding='utf-8'))
        # boxes = boxes_dict['shapes']
        # [[x1,y1],[x2,y2]], ...
        # boxes = [box['points'] for box in boxes]

        # for dog detection
        boxes = [self.bbox[index]]  # x1xy1xx2xy2

        # for every bbox
        for k in range(len(boxes)):
            # 3.1 get bbox:[x1,y1,x2,y2] and classes
            # for circle detection
            # bbox = np.array([boxes[k][0][0], boxes[k][0][1], boxes[k][1][0], boxes[k][1][1]])

            # for dog detection
            bbox = boxes[k]
            bbox = bbox.split('x')
            bbox = np.array([int(val) for val in bbox])
            cls_id = 0
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            # make sure bbox valid
            if h > 0 and w > 0:
                # 3.2 get gaussian radius by size of (h,w)
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))  # math.ceil(h) 向上取整
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                # 3.3 process hm,wh,reg
                # 3.3.1 draw gaussian on hm[class_id] with center and radius
                draw_umich_gaussian(hm[cls_id], ct_int, radius)
                # 3.3.2 kth box w,h in 'wh'
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                # 3.3.3 kth box regression
                reg[k] = ct - ct_int
                # 3.3.4 kth box regression mark
                reg_mask[k] = 1
                # apply weighted regression near center not only just apply regression on center point
                draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        label = {'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        hm_a = hm.max(axis=0, keepdims=True)
        dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
        label.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
        label.update({'reg': reg})
        return inp, label
