#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '[imgfunc] 画像のI/O関係'
#

import os
import sys
import cv2
from logging import getLogger
logger = getLogger(__name__)

[sys.path.append(d) for d in ['./Tools/', '../Tools/'] if os.path.isdir(d)]
from func import fileFuncLine, getFilePath


def getCh(ch):
    """
    入力されたチャンネル数をOpenCVの形式に変換する
    [in]  ch:入力されたチャンネル数 (type=int or np.shape)
    [out] OpenCVの形式
    """

    logger.debug('getCh({})'.format(ch))
    if(ch == 1):
        return cv2.IMREAD_GRAYSCALE
    elif(ch == 3):
        return cv2.IMREAD_COLOR
    else:
        return cv2.IMREAD_UNCHANGED


def read(path, ch=3):
    """
    画像の有無を確認して読み込む
    ※基本的にはreadN()と同じなので省略
    """

    logger.debug('read({},{})'.format(path, ch))
    if isImgPath(path):
        logger.debug('imread:\t{}'.format(path))
        return cv2.imread(path, getCh(ch))
    else:
        return None


def readN(path_list, ch=3):
    """
    画像の有無を確認して読み込む
    [in]  path: 画像のパス
    [in]  ch:   画像のチャンネル
    [out] 読み込んだ画像
    """
    logger.debug('imreadN:\t{}'.format(path_list))
    return [read(path, ch) for path in path_list if isImgPath(path)]


def write(folder, name, img, ext='.jpg'):
    """
    画像に逐次連番を追加して保存する
    [in]  folder: 保存するフォルダ
    [in]  name:   保存するファイル名
    [in]  img:    保存する画像
    [in]  ext:    拡張子
    [out] path:   保存するファイルのパス
    """

    logger.debug('write({},{},{},{})'.format(folder, name, img.shape, ext))
    write.__dict__.setdefault('count', 0)
    path = getFilePath(folder, name+str(write.count).zfill(4), ext)
    cv2.imwrite(path, img)
    write.count += 1
    logger.debug('\t count: {}'.format(write.count))
    return path


def isImgPath(path, silent=False):
    """
    入力されたパスが画像か判定する
    [in]  path:   画像か判定したいパス
    [in]  silent: cv2.imread失敗時にエラーを表示させない場合はTrue
    [out] 画像ならTrue
    """

    logger.debug('isImgPath({},{})'.format(path, silent))
    if not type(path) is str:
        return False

    import imghdr
    if not os.path.isfile(path):
        if not silent:
            logger.error('image not found: {}'.format(path))
            logger.error(fileFuncLine())

        return False

    if imghdr.what(path) is None:
        if not silent:
            logger.error('image not found: {}'.format(path))
            logger.error(fileFuncLine())

        return False
    else:
        return True
