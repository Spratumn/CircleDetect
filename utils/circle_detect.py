import numpy as np
import cv2
import matplotlib.pyplot as plt


def _sharp(gray_img):
    t1 = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])  # 定义拉普拉斯滤波器
    shp = gray_img * 1  # 设置一个新的图片变量，防止修改原图片
    shp = cv2.filter2D(shp, -1, t1)
    return gray_img+shp


def _gray_transform(gray_img, std=100):
    index_on = gray_img >= std
    index_off = gray_img < std
    gray_img[index_off] = gray_img[index_off]-50
    gray_img[index_on] = gray_img[index_on]+50
    gray_img[gray_img < 0] = 0
    gray_img[gray_img > 255] = 255
    return gray_img


def detect_circles(image, sharp=True, blur='gaussian', gray_trans=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    width, height = gray.shape
    radius = int(min(width/2, height/2))
    if sharp:
        gray = _sharp(gray)
    if gray_trans:
        gray = _gray_transform(gray)
    if blur == 'gaussian':
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
    elif blur == 'mid':
        gray = cv2.medianBlur(gray, 9)
    else:
        pass
    edge_canny = cv2.Canny(gray, 50, 150)
    # 霍夫圆变换
    circles = cv2.HoughCircles(edge_canny,
                               cv2.HOUGH_GRADIENT,
                               1, 40,
                               param1=25, param2=20,
                               minRadius=0, maxRadius=radius)
    return circles
    # 将检测的圆画出来
    # for i in circles[0, :]:
    #     cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画出外圆
    #     cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)  # 画出圆心

