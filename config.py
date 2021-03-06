import numpy as np


class Config(object):
    # system
    DATA_DIR = 'data'
    LOG_DIR = 'log'
    WEIGHTS_DIR = 'log/weights'
    OUTPUT_DIR = 'output'
    GPU = [0]
    DEVICE = 'cuda'
    NUM_WORKERS = 0  # 'dataloader threads. 0 for single-thread.'
    NOT_CUDA_BENCHMARK = False  # disable when the input size is not fixed.
    SEED = 317  # random seed from CornerNet

    # log
    PRINT_ITER = 0  # disable progress bar and print to screen
    HIDE_DATA_TIME = True  # not display time during training.
    SAVE_ALL = True  # save model to disk every 5 epochs.
    METRIC = 'loss'  # main metric to save best model
    VIS_THRESH = 0.3  # visualization threshold
    DEBUGGER_THEME = 'white'  # choices=['white', 'black']

    # model
    NUM_CLASS = 2

    ARCH = 'res_18'  # 'res_18 | res_34 | litnet'
    HEAD = {'hm': NUM_CLASS,
            'wh': 2,
            'reg': 2}
    HEAD_CONV = 64  # 0 for no conv layer', '64 for resnet'
    DOWN_RATIO = 4  # output stride
    PAD = 127 if ARCH == 'hourglass' else 31
    NUM_STACKS = 2 if ARCH == 'hourglass' else 1

    # dataset
    # input
    INPUT_SIZE = [512, 512]
    CLASS_NAME = {0: '__background__',
                  1: 'circle'}
    DATA_MEAN = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    DATA_STD = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    # train
    LR = 2.0e-5  # learning rate.
    LR_STEP = [90, 120]  # drop learning rate by 10.'
    NUM_EPOCHS = 10  # total training epochs.
    BATCH_SIZE = 16
    VAL_INTERVALS = 5  # number of epochs to run validation.

    # test
    FLIP_TEST = True  # flip data augmentation
    TEST_SCALES = [1]  # multi scale test augmentation.
    NMS = True  # run nms in testing
    K = 20  # max number of output objects
    CENTER_THRESH = 0.2  # threshold for detection score.

    # data augment
    RAND_CROP = True  # use the random crop data augmentation
    SHIFT = 0.1  # when not using random crop apply shift augmentation.
    SCALE = 0.4  # when not using random crop apply scale augmentation.
    ROTATE = 0  # when not using random crop apply rotation augmentation.
    FLIP = 0.5  # probability of applying flip augmentation.
    COLOR_AUG = False  # not use the color augmentation

    # loss
    USE_OFFSET = True
    OFFSET_LOSS = 'l2'  # regression loss: sl1 | l1 | l2
    HM_WEIGHT = 1  # loss weight for key point heat maps
    OFF_WEIGHT = 1  # loss weight for key point local offsets
    WH_WEIGHT = 2  # loss weight for bounding box size

    NORM_WH = True  # 'L1(\hat(y) / y, 1) or L1(\hat(y), y)
    DENSE_WH = True  # apply weighted regression near center or just apply regression on center point




