#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '[imgfunc] 画像の貼り付け'
#

import os
import sys
import cv2
import numpy as np
from logging import getLogger
logger = getLogger(__name__)

[sys.path.append(d) for d in ['./Tools/', '../Tools/'] if os.path.isdir(d)]
import imgfunc.convert_img as CNV


def paste(fg, bg, rot=0, x=0, y=0, mask_flg=True, rand_rot_flg=True, rand_pos_flg=True):
    """
    背景に前景を重ね合せる
    [in]  fg:           重ね合せる前景
    [in]  bg:           重ね合せる背景
    [in]  rot:          重ね合わせ時の前景回転角
    [in]  x:            重ね合わせ時の前景x位置
    [in]  y:            重ね合わせ時の前景y位置
    [in]  mask_flg:     マスク処理を大きめにするフラグ
    [in]  rand_rot_flg: 前景をランダムに回転するフラグ
    [in]  rand_pos_flg: 前景をランダムに配置するフラグ
    [out] 重ね合せた画像
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html#bitwise-operations
    """

    logger.debug('paste({},{},{},{},{},{},{},{})'.format(
        fg.shape, bg.shape, rot, x, y, mask_flg, rand_rot_flg, rand_pos_flg
    ))

    # 画素の最大値
    max_val = 255

    # Load two images
    img1 = bg.copy()
    white = (max_val, max_val, max_val)
    angle = [-90, 90]  # ランダム回転の範囲
    scale = 1.0  # 画像の拡大率
    logger.debug('\trand_rot_flg: {}'.format(rand_rot_flg))
    if rand_rot_flg:
        # ランダムに回転
        img2, rot = CNV.rotateR(fg, angle, scale, white)
    else:
        # 任意の角度で回転
        img2 = CNV.rotate(fg, rot, scale, white)

    # I want to put logo on top-left corner, So I create a ROI
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]
    logger.debug('\trand_pos_flg: {}'.format(rand_pos_flg))
    if rand_pos_flg:
        x = np.random.randint(0, w1 - w2 + 1)
        y = np.random.randint(0, w1 - w2 + 1)

    roi = img1[x:x + w2, y:y + h2]
    logger.debug('\trot:{}, pos:({},{}), shape:{}'.format(
        rot, x, y, roi.shape))

    def masked(img):
        logger.debug('masked({})'.format(img.shape))
        if len(img.shape) < 3:
            return False
        elif img.shape[2] != 4:
            return False
        else:
            return True

    # Now create a mask of logo and create its inverse mask also
    if not masked(img2):
        thresh = 10
        mask = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(
            cv2.bitwise_not(mask), thresh, max_val, cv2.THRESH_BINARY
        )
    else:
        mask = img2[:, :, 3]

    thresh = 200
    ret, mask_inv = cv2.threshold(
        cv2.bitwise_not(mask), thresh, max_val, cv2.THRESH_BINARY
    )

    if mask_flg:
        kernel1 = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((3, 3), np.uint8)
        mask_inv = cv2.dilate(mask_inv, kernel1, iterations=1)
        mask_inv = cv2.erode(mask_inv, kernel2, iterations=1)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[x:x + w2, y:y + h2] = dst
    logger.debug('\timg1.shape: {}'.format(img1.shape))
    return img1, (rot, x, y)
