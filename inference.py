import numpy as np
import torch
import cv2

from models.decode import decode
from models.utils import flip_tensor
from utils.image_process import get_affine_transform
from utils.post_process import post_process
from models.model import create_model, load_model
from utils.nms import nms
from utils.circle_detect import detect_circles
from config import Config
import matplotlib.pyplot as plt


class Detector:
    def __init__(self, model_path, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        print('Creating model...')
        self.model = create_model(self.cfg, 'res_18')
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(self.cfg.DEVICE)
        self.model.eval()

        self.mean = np.array(self.cfg.DATA_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.cfg.DATA_STD, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = cfg.K
        self.scales = self.cfg.TEST_SCALES
        self.pause = True

    def pre_process(self, image, scale):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)

        inp_height = (new_height | self.cfg.PAD) + 1
        inp_width = (new_width | self.cfg.PAD) + 1

        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(resized_image, trans_input,
                                   (inp_width, inp_height),
                                   flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

        # flip image by width axes then concat two image as batch_size = 2
        if self.cfg.FLIP_TEST:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.cfg.DOWN_RATIO,
                'out_width': inp_width // self.cfg.DOWN_RATIO}
        return images, meta

    def process(self, images):
        with torch.no_grad():
            output = self.model(images)
            hm = output['hm'].softmax(dim=1)
            wh = output['wh']
            if self.cfg.USE_OFFSET:
                offset = output['offset']
            else:
                offset = None

            if self.cfg.FLIP_TEST:
                # flip hm[1] then get mean(hm[0], hm[1])
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                # flip wh[1] then get mean(wh[0], wh[1])
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                if self.cfg.USE_OFFSET:
                    offset = (offset[0:1] + flip_tensor(offset[1:2])) / 2
            hm = hm[:, 1:, :]
            dets = decode(hm, wh, offset, K=self.cfg.K)
            # dets: shape of [N,K,6]. det:[x1,y1,x2,y2,score,class_id] in down_sample size
        return dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        # [N,K,6] -> [1, N*K, 6]
        dets = dets.reshape(-1, dets.shape[2])
        dets = post_process(dets.copy(),
                            [meta['c']],
                            [meta['s']],
                            meta['out_height'],
                            meta['out_width'],
                            self.cfg.NUM_CLASS,
                            score_thresh=self.cfg.CENTER_THRESH)

        for j in range(1, self.cfg.NUM_CLASS):
            dets[j] = np.array(dets[j], dtype=np.float32).reshape(-1, 5)
            dets[j][:, :4] /= scale
        return dets

    def merge_outputs(self, detections):
        # detections: list of dets, dets: detection dict{1:det_array,2:det_array...}. det_array: shape of [k,5]
        # return: {1:det_array,2:det_array...}. det_array: shape of [k,5]
        res_dets = {}
        for j in range(1, self.cfg.NUM_CLASS):
            res_dets[j] = np.concatenate(
                [dets[j] for dets in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.cfg.NMS:
                res_index = nms(res_dets[j], 0.5)
                res_dets[j] = res_dets[j][res_index]

        scores = np.hstack([res_dets[j][:, 4] for j in range(1, self.cfg.NUM_CLASS)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.cfg.NUM_CLASS):
                keep_inds = (res_dets[j][:, 4] >= thresh)
                res_dets[j] = res_dets[j][keep_inds]
        return res_dets

    def draw_results(self, image, result, max_per_class=20):
        # result: {1:det_array,2:det_array...}. det_array: shape of [k,5]

        class_id_name = self.cfg.CLASS_NAME
        img = image
        for i in range(1, self.cfg.NUM_CLASS):
            rets = result[i]
            obj_count = min(rets.shape[0], max_per_class)
            for j in range(obj_count):
                text_str = class_id_name[i] + str(round(rets[j][4], 3))
                cv2.putText(img, text_str, (rets[j][0], int(rets[j][1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                cv2.rectangle(img,
                              (rets[j][0], rets[j][1]), (rets[j][2], rets[j][3]),
                              (0, 0, 255), 2)
        return img

    def run(self, image_path, draw_result=False):
        # return: {1:det_array,2:det_array...}. det_array: shape of [k,5]
        image = cv2.imread(image_path)
        detections = []
        for scale in self.scales:
            images, meta = self.pre_process(image, scale)
            images = images.to(self.cfg.DEVICE)
            # dets: shape of [N,K,6]. det:[x1,y1,x2,y2,score,class_id]
            dets = self.process(images)
            # dets: detection dict{1:det_array,2:det_array...}. det_array: shape of [k,5]
            dets = self.post_process(dets, meta, scale)
            detections.append(dets)
        result = self.merge_outputs(detections)
        if draw_result:
            image = self.draw_results(image, result, max_per_class=self.max_per_image)
            image_name = image_path.split('/')[-1]
            output_path = 'output/image test/' + image_name
            cv2.imwrite(output_path, image)

        return result

    def show_hm(self, image_path):
        image = cv2.imread(image_path)
        images, meta = self.pre_process(image, 1)
        images = images.to(self.cfg.DEVICE)
        output = self.model(images)
        hm = output['hm'][0]
        hm = hm.softmax(dim=0)
        hm = hm.cpu().detach().numpy()
        print(hm.shape)

        plt.subplot(1, 2, 1)
        plt.imshow(hm[0])
        plt.subplot(1, 2, 2)
        plt.imshow(hm[1])
        plt.show()


if __name__ == '__main__':
    import pandas as pd
    import time
    cfg = Config()

    start = time.time()
    detecter = Detector('log/weights/model_last_res_34.pth', cfg)
    print('load model cost: ', time.time()-start)
    image_path = 'data/test_images/moulde.jpg'
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    results = detecter.run(image_path)
    bboxes = results[1]
    bboxes[:, 0] = np.maximum(0, bboxes[:, 0] - 5)
    bboxes[:, 1] = np.maximum(0, bboxes[:, 1] - 5)
    bboxes[:, 2] = np.minimum(w, bboxes[:, 2] + 5)
    bboxes[:, 3] = np.minimum(h, bboxes[:, 3] + 5)
    for j in range(bboxes.shape[0]):

        cv2.rectangle(image,
                      (bboxes[j][0], bboxes[j][1]), (bboxes[j][2], bboxes[j][3]),
                      (0, 0, 255), 2)
    circles = []
    for bbox in bboxes:
        bbox = [int(val) for val in bbox[0:4]]
        xc = (bbox[0] + bbox[2]) / 2
        yc = (bbox[1] + bbox[3]) / 2
        h = (bbox[3] - bbox[1])
        w = (bbox[2] - bbox[0])
        radius = min(h, w) // 5
        roi_img = image[bbox[1]: bbox[3], bbox[0]: bbox[2], :]
        try:
            c = detect_circles(roi_img, gray_trans=True)[0][0]
            circles.append([int((c[0] + bbox[0]) * 0.7 + xc * 0.3), int((c[1] + bbox[1]) * 0.7 + yc * 0.3), int(c[2])])
        except TypeError:
            circles.append([int(xc), int(yc), radius])

    for c in circles:
        cv2.circle(image, (c[0], c[1]), c[2], (0, 255, 0), 2)  # 画出外圆
        cv2.circle(image, (c[0], c[1]), 2, (0, 0, 255), 2)  # 画出圆心
    cv2.imshow(image_path, image)
    cv2.waitKey(0)
    image_name = image_path.split('/')[-1]
    output_path = 'output/image test/' + image_name
    cv2.imwrite(output_path, image)



    # start = time.time()
    # for i in range(5):
    #     image_path = 'data/test_images/00{}.jpg'.format(i)
    #     detecter.run(image_path, draw_result=True)
    # print((time.time()-start)/7)



