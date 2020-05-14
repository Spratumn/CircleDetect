import numpy as np
import torch
import cv2

from models.decode import decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import post_process
from utils.debugger import Debugger
from models.model import create_model, load_model
from utils.nms import nms
from config import Config


class Detector:
    def __init__(self, model_path):
        super(Detector, self).__init__()
        self.cfg = Config()
        print('Creating model...')
        self.model = create_model(self.cfg)
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(self.cfg.DEVICE)
        self.model.eval()

        self.mean = np.array(self.cfg.DATA_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.cfg.DATA_STD, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
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
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

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
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            if self.cfg.FLIP_TEST:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1]
            dets = decode(hm, wh, reg=reg, cat_spec_wh=self.cfg.CAT_SPEC_WH, K=self.cfg.K)
        return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.cfg.NUM_CLASS)
        for j in range(1, self.cfg.NUM_CLASS + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.cfg.NUM_CLASS + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.cfg.NMS:
                res_index = nms(results[j], 0.5)
                results[j] = results[j][res_index]
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.cfg.NUM_CLASS + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.cfg.NUM_CLASS + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def show_results(self, debugger, image, results):
        debugger.add_img(image)
        for j in range(1, self.cfg.NUM_CLASS + 1):
            for bbox in results[j]:
                if bbox[4] > self.cfg.VIS_THRESH:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4])
        debugger.show_all_imgs(pause=self.pause)

    def run(self, image_path):
        debugger = Debugger(num_classes=1)
        image = cv2.imread(image_path)
        detections = []
        for scale in self.scales:
            images, meta = self.pre_process(image, scale)
            images = images.to(self.cfg.DEVICE)
            output, dets = self.process(images)
            dets = self.post_process(dets, meta, scale)
            detections.append(dets)
        results = self.merge_outputs(detections)
        # self.show_results(debugger, image, results)
        print(len(results[1]))
        bbox = results[1][5][0:4]
        print(bbox)

        return {'results': results}


if __name__ == '__main__':
    # detecter = Detector('log/weights/model_last.pth')
    # result = detecter.run('data/dataset/great_pyrenees_153.jpg')
    # print(result['results'])
    img = cv2.imread('data/dataset/great_pyrenees_153.jpg')
    cv2.rectangle(img,(370,134),(371,135),(0,0,255),2)
    cv2.imshow('',img)
    cv2.waitKey(0)

