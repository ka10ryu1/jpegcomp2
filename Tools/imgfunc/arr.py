#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '[imgfunc] 行列と画像の変換'
#

import os
import sys
import cv2
import numpy as np
from logging import getLogger
logger = getLogger(__name__)

try:
    import cupy as xp
except ImportError:
    logger.warning('not import cupy')

[sys.path.append(d) for d in ['./Tools/', '../Tools/'] if os.path.isdir(d)]
from func import fileFuncLine
import imgfunc.convert_img as CNV


def arr2x(arr, flg=cv2.INTER_NEAREST):
    """
    行列を画像に変換し、サイズを2倍にする
    [in] arr: 2倍にする行列
    [in] flg: 2倍にする時のフラグ
    [out] 2倍にされた行列
    """

    logger.debug('arr2x({},{})'.format(arr.shape, flg))
    return imgs2arr(CNV.size2x(arr2imgs(arr), flg))


def arrNx(arr, rate, flg=cv2.INTER_NEAREST):
    """
    行列を画像に変換し、サイズをN倍にする
    [in] arr:  N倍にする行列
    [in] rate: 倍率
    [in] flg:  N倍にする時のフラグ
    [out] N倍にされた行列
    """

    # logger.debug('arrNx({},{},{})'.format(arr.shape, rate, flg))
    if(len(arr.shape) == 3):
        img = arr2img(arr)
        return img2arr(CNV.resize(img, rate, flg))

    if(len(arr.shape) == 4):
        imgs = arr2imgs(arr)
        return imgs2arr(CNV.resizeN(imgs, rate, flg))


def img2arr(img, norm=255, dtype=np.float32, gpu=-1):
    """
    入力画像をChainerで利用するために変換する
    ※詳細はimgs2arrとほぼ同じなので省略
    """

    # logger.debug('img2arr({},{},{},{})'.format(
    # img.shape, norm, dtype.__name__, gpu
    # ))
    try:
        w, h, _ = img.shape
    except:
        w, h = img.shape[:2]

    if(gpu >= 0):
        return xp.array(img, dtype=dtype).reshape((-1, w, h)) / norm
    else:
        return np.array(img, dtype=dtype).reshape((-1, w, h)) / norm


def imgs2arr(imgs, norm=255, dtype=np.float32, gpu=-1):
    """
    入力画像リストをChainerで利用するために変換する
    [in]  imgs:  入力画像リスト
    [in]  norm:  正規化する値（255であれば、0-255を0-1に正規化する）
    [in]  dtype: 変換するデータタイプ
    [in]  gpu:   GPUを使用する場合はGPUIDを入力する
    [out] 生成された行列
    """

    # logger.debug('imgs2arr(N={},{},{},{})'.format(
    #     len(imgs), norm, dtype.__name__, gpu
    # ))
    try:
        w, h, ch = imgs[0].shape
    except:
        w, h = imgs[0].shape
        ch = 1

    if(gpu >= 0):
        return xp.array(imgs, dtype=dtype).reshape((-1, ch, w, h)) / norm
    else:
        return np.array(imgs, dtype=dtype).reshape((-1, ch, w, h)) / norm


def arr2img(arr, norm=255, dtype=np.uint8):
    """
    Chainerの出力をOpenCVで可視化するために変換する入力（単画像用）
    ※詳細はarr2imgsとほぼ同じなので省略
    """

    try:
        ch, h, w = arr.shape
    except:
        h, w = arr.shape
        ch = 1

    y = np.array(arr).reshape((h, w, ch)) * norm
    # logger.debug('arr2img({},{},{})->{}'.format(
    #     arr.shape, norm, dtype.__name__, y.shape
    # ))
    return np.array(y, dtype=dtype)


def arr2imgs(arr, norm=255, dtype=np.uint8):
    """
    Chainerの出力をOpenCVで可視化するために変換する（画像リスト用）
    [in]  arr:   Chainerから出力された行列
    [in]  norm:  正規化をもとに戻す数（255であれば、0-1を0-255に変換する）
    [in]  dtype: 変換するデータタイプ
    [out] OpenCV形式の画像に変換された行列
    """

    try:
        ch, size = arr.shape[1], arr.shape[2]
    except:
        logger.error('input data is not img arr')
        logger.error(fileFuncLine())
        exit(1)

    y = np.array(arr).reshape((-1, size, size, ch)) * norm
    # logger.debug('arr2imgs({},{},{})->{}'.format(
    #     arr.shape, norm, dtype.__name__, y.shape
    # ))
    return np.array(y, dtype=dtype)
