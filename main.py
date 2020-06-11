from utils.make_list import make_datalist
import numpy as np
from utils.metrics import compute_metrics
from config import Config
from utils.dataset import TrainCircleDataset
import matplotlib.pyplot as plt
from inference import Detector
from models.centernet import CenterNet
if __name__ == '__main__':
    # anns = np.load('data/dataset/augment data/train_anns.npy', allow_pickle=True).item()
    # image_paths = anns['image_paths']
    # bboxs = anns['bboxs']  # [x1,y1,x2,y2]
    # print(len(bboxs))
    # print(image_paths[1])
    # print(bboxs[1])

    # gt_bbox = np.array([[77, 134, 115, 171], [242, 285, 278, 321], [442, 185, 481, 228], [465, 130, 511, 180]])
    # pre_bbox = np.array([[7.7834442e+01, 1.3114578e+02, 1.1509906e+02, 1.6780325e+02, 6.4712548e-01],
    #                      [2.3694418e+02, 2.8126999e+02, 2.7572971e+02, 3.1951279e+02, 6.1400807e-01],
    #                      [4.3898438e+02, 1.8438614e+02, 4.7392853e+02, 2.1855090e+02, 5.9599435e-01],
    #                      [4.6116867e+02, 1.2944656e+02, 5.0793793e+02, 1.7586459e+02, 2.8655225e-01]])
    #
    # print(compute_metrics(pre_bbox, gt_bbox))
    cfg = Config()
    # det = Detector('log/weights/model_last.pth', cfg)
    # det.show_hm('data/test_images/002.jpg')
    # ct = CenterNet(cfg)
    # from torchsummary import summary
    # summary(ct,(3, 224,224),device='cpu')

    # dst = TrainCircleDataset(cfg)
    # inp, label = dst.__getitem__(3)
    # print(label['dense_wh'].shape)
    # plt.subplot(1, 2, 1)
    # plt.imshow(label['dense_wh'][0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(label['dense_wh'][1])
    # print(label['wh'])
    # plt.show()






