import cv2
import json
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia


def get_bbox_from_json(json_file):
    """ get boxes[x1,y1,x2,y2]  with center and w,h from json file"""
    boxes_dict = json.load(open(json_file, encoding='utf-8'))
    boxes = boxes_dict['shapes']
    # [[x1,y1],[x2,y2]], ...
    boxes = [box['points'] for box in boxes]
    # boxes[0][0][0]  # x1
    # boxes[0][0][1]  # y1
    # boxes[0][1][0]  # x2
    # boxes[0][1][1]  # y2
    boxes = [[int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])]
             for box in boxes]
    return boxes


def get_crop_dataset(augmenter, image, bboxs, max_h, max_w):
    temp_aug_bbox = []
    for bbox in bboxs:
        temp_aug_bbox.append(ia.BoundingBox(x1=bbox[0], 
                                            x2=bbox[2], 
                                            y1=bbox[1], 
                                            y2=bbox[3]))
    bbs = ia.BoundingBoxesOnImage(temp_aug_bbox, shape=image.shape)

    # keep same aug
    seq_det = augmenter.to_deterministic()
    
    res_img = seq_det.augment_image(image)
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    res_bboxs = []
    for bbx in bbs_aug:
        if bbx.x1 >= 0 and bbx.y1 >= 0 and bbx.x2 < max_w and bbx.y2 < max_h:
            res_bboxs.append([int(bbx.x1), int(bbx.y1),  int(bbx.x2),  int(bbx.y2)])
    return res_img, res_bboxs


if __name__ == '__main__':
    # img = cv2.imread('0.jpg')
    # bboxs = get_bbox_from_json('0.json')
    #
    # aug = iaa.CropToFixedSize(width=512, height=512)
    # res_img, bbox = get_crop_dataset(aug, img, bboxs, 512, 512)
    #
    # for box in bbox:
    #     cv2.rectangle(res_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
    # cv2.imshow('', res_img)
    # cv2.waitKey(0)
    data_set_dir = '../data/dataset/'
    augment_dataset_dir = 'images/'
    res_img_paths = []
    res_bboxs = []

    augment_rate = 40
    crop_aug = iaa.Sequential([iaa.OneOf([
                            iaa.CropToFixedSize(width=64, height=64),
                            iaa.CropToFixedSize(width=128, height=128),
                            iaa.CropToFixedSize(width=192, height=192),
                            iaa.CropToFixedSize(width=256, height=256),
                            iaa.CropToFixedSize(width=320, height=320),
                            iaa.CropToFixedSize(width=384, height=384),
                            iaa.CropToFixedSize(width=448, height=448),
                            iaa.CropToFixedSize(width=512, height=512)])])
    resize_aug = iaa.Resize({"height": 512, "width": 512})

    aug = iaa.Sequential([crop_aug, resize_aug])

    for i in range(346):
        src_img = cv2.imread(data_set_dir + '{}.jpg'.format(i))
        src_bboxs = get_bbox_from_json(data_set_dir + '{}.json'.format(i))
        for j in range(augment_rate):
            res_img, bbox = get_crop_dataset(aug, src_img, src_bboxs, 512, 512)
            if len(bbox) > 0:
                image_path = augment_dataset_dir + '{0}_{1}.jpg'.format(i, j)
                res_img_paths.append('data/dataset/augment data/images/' + '{0}_{1}.jpg'.format(i, j))
                res_bboxs.append(bbox)
                # for box in bbox:
                #     cv2.rectangle(res_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
                # cv2.imshow(image_path, res_img)
                # cv2.waitKey(0)
                cv2.imwrite(image_path, res_img)

    res_anns = {'image_paths': res_img_paths, 'bboxs': res_bboxs}
    np.save('anns.npy', res_anns)





