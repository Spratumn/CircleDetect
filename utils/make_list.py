import numpy as np
import random

DATA_DIR = 'data/dataset/augment data/'
DATA_PARTITION = [0.9, 0.95]


def make_datalist():
    anns = np.load(DATA_DIR + 'anns.npy', allow_pickle=True).item()
    image_paths = anns['image_paths']
    bboxs = anns['bboxs']  # [x1,y1,x2,y2]
    total_length = len(image_paths)
    # random samples
    index = [i for i in range(total_length)]
    random.shuffle(index)
    image_paths = [image_paths[i] for i in index]
    bboxs = [bboxs[i] for i in index]
    # set partition size
    eval_part = int(total_length * DATA_PARTITION[0])
    test_part = int(total_length * DATA_PARTITION[1])
    #

    train_dataset_image_paths = image_paths[:eval_part]
    train_dataset_bboxs = bboxs[:eval_part]
    eval_dataset_image_paths = image_paths[eval_part:test_part]
    eval_dataset_bboxs = bboxs[eval_part:test_part]
    test_dataset_image_paths = image_paths[test_part:]
    test_dataset_bboxs = bboxs[test_part:]

    train_anns = {'image_paths': train_dataset_image_paths, 'bboxs': train_dataset_bboxs}
    eval_anns = {'image_paths': eval_dataset_image_paths, 'bboxs': eval_dataset_bboxs}
    test_anns = {'image_paths': test_dataset_image_paths, 'bboxs': test_dataset_bboxs}
    np.save(DATA_DIR + 'train_anns.npy', train_anns)
    np.save(DATA_DIR + 'eval_anns.npy', eval_anns)
    np.save(DATA_DIR + 'test_anns.npy', test_anns)


if __name__ == '__main__':
    make_datalist()
