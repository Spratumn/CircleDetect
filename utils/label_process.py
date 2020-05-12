import numpy as np
import json
import cv2


def get_bbox(json_file):
    """ get circle boxes with center and w,h"""
    boxes_dict = json.load(open(json_file, encoding='utf-8'))
    boxes = boxes_dict['shapes']
    img_data = boxes_dict['imageData']
    img_path = boxes_dict['imagePath']
    img_size = [boxes_dict['imageHeight'], boxes_dict['imageWidth']]
    print(img_path)
    print(img_size)
    # [[x1,y1],[x2,y2]], ...
    boxes = [box['points'] for box in boxes]
    # boxes[0][0][0]  # x1
    # boxes[0][0][1]  # y1
    # boxes[0][1][0]  # x2
    # boxes[0][1][1]  # y2
    boxes = [[((box[0][0] + box[1][0])/2, (box[0][1] + box[1][1])/2),  # [xc,yc]
              box[1][0] - box[0][0],  # w
              box[1][1] - box[0][1]  # h
              ]
             for box in boxes]

    return img_data, boxes


if __name__ == '__main__':
    json_path = '../data/mould02.json'
    data, boxes = get_bbox(json_path)


