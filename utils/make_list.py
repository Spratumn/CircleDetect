import os
import pandas as pd
from sklearn.utils import shuffle  # random

"""
将训练集中的标注数据的路径打乱顺序保存到csv
"""

CSV_DIR = "../data"
DATA_DIR = "../data/dataset"
DATA_PARTITION = [0.8, 0.9]


def get_data_dir():
    image_list = []
    json_list = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            json_list.append(os.path.join(DATA_DIR[3:], filename))
        elif filename.endswith('.jpg'):
            image_list.append(os.path.join(DATA_DIR[3:], filename))
    assert len(image_list) == len(json_list)
    return image_list, json_list


def make_datalist():
    image_list, json_list = get_data_dir()
    # set partition size
    total_length = len(json_list)
    eval_part = int(total_length * DATA_PARTITION[0])
    test_part = int(total_length * DATA_PARTITION[1])
    #
    all_data = pd.DataFrame({'image': image_list, 'json': json_list})
    all_shuffle = shuffle(all_data)

    train_dataset = all_shuffle[:eval_part]
    eval_dataset = all_shuffle[eval_part:test_part]
    test_dataset = all_shuffle[test_part:]

    train_dataset.to_csv(CSV_DIR + "/train.csv", index=False)
    eval_dataset.to_csv(CSV_DIR + "/eval.csv", index=False)
    test_dataset.to_csv(CSV_DIR + "/test.csv", index=False)


if __name__ == '__main__':
    make_datalist()
