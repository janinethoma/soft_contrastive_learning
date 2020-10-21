import math

import cv2
import numpy as np


def resize_img(img, max_size):
    scale = max_size / float(max(img.shape[0], img.shape[1]))
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)


def standard_size(img, h=180, w=240):
    ih = img.shape[0]
    iw = img.shape[1]
    scale = max(h / ih, w / iw)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    nh = img.shape[0]
    nw = img.shape[1]
    return img[math.floor((nh - h) / 2.0):math.floor((nh - h) / 2.0) + h,
           math.floor((nw - w) / 2.0):math.floor((nw - w) / 2.0) + w, :]


def put_text(text, image, scale=1, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left = (10, 35)
    line_type = 2
    return cv2.putText(image, text, bottom_left, font, scale, color, line_type)


def merge_images(left_image, right_image):
    right_image = cv2.resize(right_image, (right_image.shape[1] * left_image.shape[0] // right_image.shape[0],
                                           left_image.shape[0]))
    img = np.concatenate((left_image, right_image), axis=1)
    return img
