import itertools
import math
import os
import random
import sys
import numpy as np
import cv2
import uuid
from PIL import Image, ImageFilter, ImageEnhance


def resample(img, resizeRange):
    resizer = random.randint(0, resizeRange)
    if resizer > 0:
        h, w = img.shape[0:2]
        img = cv2.resize(img, (img.shape[1] // resizer, img.shape[0] // resizer))
        img = cv2.resize(img, (w, h))
    return img


def jittering_blur(img, max_sigma):
    kernel_list = [3, 5, 7, 11]
    kernel = random.choice(kernel_list)
    sigma = random.uniform(1, max_sigma)
    return cv2.GaussianBlur(img, (kernel, kernel), sigma)


def jittering_color(img, h1=100, h2=110, s1=90, s2=100, v1=90, v2=100):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)

    random_h_scale = random.randint(h1, h2) / 100.0
    random_s_scale = random.randint(s1, s2) / 100.0
    random_v_scale = random.randint(v1, v2) / 100.0

    hsv[:, :, 0] *= random_h_scale
    hsv[:, :, 1] *= random_s_scale
    hsv[:, :, 2] *= random_v_scale

    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def jittering_border(image, max_x_percent=2, max_y_percent=10):
    scale_x = random.randint(0, max_x_percent) / 100.0
    scale_y = random.randint(0, max_y_percent) / 100.0
    height, width = image.shape[:2]
    border_x = int(width * scale_x)
    border_y = int(height * scale_y)

    roi = image[border_y: height - border_y, border_x:width - border_x]
    return roi


def jittering_scale(image, min_scale=0.5, max_scale=1.0):
    h, w = image.shape[:2]

    scale = random.uniform(min_scale, max_scale)
    scaled_h = int(h * scale)
    scaled_w = int(w * scale)

    image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

    return image


def random_rank_blur(img, rank_blur_range):
    img = Image.fromarray(img)
    w, h = img.size[0], img.size[1]
    img = img.resize((2 * w, 2 * h))
    rank = np.random.randint(0, rank_blur_range * rank_blur_range)
    img = img.filter(ImageFilter.RankFilter(rank_blur_range, rank))
    img = img.resize((w, h))
    img = np.array(img)
    return img


def random_motion_blur(img, motion_blur_range):
    img_array = img
    angle = np.random.randint(0, 360)
    motion_blur_range = np.random.randint(1, motion_blur_range)
    M = cv2.getRotationMatrix2D((motion_blur_range / 2, motion_blur_range / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(motion_blur_range))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (motion_blur_range, motion_blur_range))

    motion_blur_kernel = motion_blur_kernel / motion_blur_range
    img_array = cv2.filter2D(img_array, -1, motion_blur_kernel)

    cv2.normalize(img_array, img_array, 0, 255, cv2.NORM_MINMAX)
    # img = Image.fromarray(img_array)
    img = img_array
    return img


def random_brightness(img, brightness_jitter):
    img = Image.fromarray(img)
    brightness_jitter_need = np.random.uniform(-brightness_jitter, brightness_jitter)
    img_brightness = ImageEnhance.Brightness(img)
    img = img_brightness.enhance(1 + brightness_jitter_need)
    img = np.array(img)
    return img

