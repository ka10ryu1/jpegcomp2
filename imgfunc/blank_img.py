#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '[imgfunc] 単色画像の生成'
#

import os
import sys
import numpy as np
from logging import getLogger
logger = getLogger(__name__)

[sys.path.append(d) for d in ['./Tools/', '../Tools/'] if os.path.isdir(d)]
from func import fileFuncLine


def white(w, h=None, ch=None):
    """
    単色（白）を生成する
    """

    if isinstance(w, tuple):
        return blank(w, 255)

    return blank((w, h, ch), 255)


def black(w, h=None, ch=None):
    """
    単色（黒）を生成する
    """

    if isinstance(w, tuple):
        return blank(w, 0)

    return blank((w, h, ch), 0)


def blank(size, color, dtype=np.uint8, min_val=0, max_val=255):
    """
    単色画像を生成する
    [in]  size: 生成する画像サイズ [h,w,ch]（chがない場合は1を設定）
    [in]  color: 色（intでグレー、tupleでカラー）
    [in]  dtype: データ型
    [in]  min_val: 色の最小値
    [in]  max_val: 色の最大値
    [out] img:   生成した単色画像
    """

    logger.debug('blank({},{},{},{},{})'.format(
        size, color, dtype.__name__, min_val, max_val
    ))
    # サイズに負数がある場合はエラー
    if np.min(size) < 0:
        logger.error('\tsize < 0: {}'.format(size))
        logger.error(fileFuncLine())
        exit(1)

    # サイズに縦横しか含まれていない場合はチャンネル追加
    if len(size) == 2:
        logger.debug('\tsize len = 2: {}'.format(size))
        size = (size[0], size[1], 1)

    if size[2] == 1:
        logger.debug('\tch = 1: {}'.format(size))
        size = (size[0], size[1])  # , 1)

    # 色がintの場合はグレースケールとして塗りつぶす
    # 0 < color < 255の範囲にない場合は丸める
    if type(color) is int:
        img = np.zeros(size, dtype=dtype)
        if color < min_val:
            color = min_val
        elif color > max_val:
            color = max_val

        logger.debug('\t0 < color < 255: {}', color)
        img.fill(color)
        return img

    # チャンネルが3じゃない時は3にする
    if len(color) == 3 and len(size) < 3:
        logger.debug('\tsize len != 3: {}'.format(size))
        size = (size[0], size[1], 3)

    img = np.zeros(size, dtype=dtype)
    img[:, :, :] = color
    return img
