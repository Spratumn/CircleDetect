import pandas as pd
import numpy as np
import cv2
import os
import math
import random
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa

from utils.image_process import get_affine_transform, affine_transform, color_aug
from utils.label_process import get_gaussian_radius, gaussian2D, draw_dense_wh, draw_gaussian


class MyDataset(Dataset):
    """
    load images and labels
    images: add augment
    labels: boxes:[[x1,y1],[x2,y2]]-->[[xc,yc],w,h]-->gaussian
    x->w;y->h
    """
    def __init__(self, cfg, phase):
        super(MyDataset, self).__init__()
        # read csv as data index
        self.cfg = cfg
        # for circle detection
        anns = np.load('data/dataset/augment data/anns.npy', allow_pickle=True).item()
        self.image_paths = anns['image_paths']
        self.bboxs = anns['bboxs']  # [x1,y1,x2,y2]
        self.num_samples = len(self.image_paths)
        # random samples
        index = [i for i in range(self.num_samples)]
        random.shuffle(index)
        self.image_paths = [self.image_paths[i] for i in index]
        self.bboxs = [self.bboxs[i] for i in index]
        self.class_id = 1
        self.max_objs = 20

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.phase = phase
        self.transformer = transforms.Compose([ImageAug()])

        print('Loaded {} samples'.format(self.num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        # 1.source image process
        # 1.1 load image by index
        img = cv2.imread(img_path)
        bboxes = self.bboxs[index]
        sample = {'image': img,
                  'bboxs': bboxes}
        img, bboxes = self.transformer(sample)['image'], self.transformer(sample)['bboxs']

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
        ind = np.zeros((self.max_objs,), dtype=np.int64)

        offset = np.zeros((self.max_objs, 2), dtype=np.float32)
        offset_mask = np.zeros((self.max_objs,), dtype=np.uint8)

        # 3.gt label process
        gt_det = []
        # for circle detection

        class_id = self.class_id
        for k in range(len(bboxes)):
            # 3.1 get bbox:[x1,y1,x2,y2] and classes
            # for circle detection
            bbox = np.array(bboxes[k])

            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            # transform same as input image
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            # make sure bbox valid
            if h > 0 and w > 0:
                # 3.2 get gaussian radius by size of (h,w)
                radius = get_gaussian_radius(math.ceil(h), math.ceil(w))  # math.ceil(h) 向上取整
                diameter = 2 * radius + 1
                gaussian = gaussian2D((diameter, diameter), diameter / 6)
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                # 3.3 process hm,wh,offset
                # 3.3.1 kth box w,h in 'wh'
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]  # 2D->1D index value of center
                if self.cfg.USE_OFFSET:
                    # 3.3.2 kth box offset
                    offset[k] = ct - ct_int
                    # 3.3.3 kth box offset mark
                    offset_mask[k] = 1
                # 3.3.4 draw dense_wh first
                draw_dense_wh(dense_wh, hm.max(axis=0), ct_int, wh[k], gaussian, radius)
                # 3.3.5 draw gaussian on hm[class_id] with center and radius
                draw_gaussian(hm[class_id], ct_int, gaussian, radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, 0])

        hm_a = hm.max(axis=0, keepdims=True)
        dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
        label = {'hm': hm, 'ind': ind, 'wh': wh, 'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask}
        if self.cfg.USE_OFFSET:
            label.update({'offset': offset, 'offset_mask': offset_mask})
        return inp, label


# imgaug Augmentation
class ImageAug(object):
    def __call__(self, sample):
        image, bboxs = sample['image'], sample['bboxs']
        if np.random.uniform(0, 1) > 0.5:
            seq = iaa.Sequential([iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
                iaa.GaussianBlur(sigma=(0, 1.0))])])
            image = seq.augment_image(image)
        return {'image': image,
                'bboxs': bboxs}


def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

