#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '[imgfunc] 画像処理部'
#

import os
import sys
import cv2
import numpy as np
from logging import getLogger
logger = getLogger(__name__)

[sys.path.append(d) for d in ['./Tools/', '../Tools/'] if os.path.isdir(d)]
from func import fileFuncLine
from imgfunc.read_write import getCh
from imgfunc.blank_img import black


def cleary(img, clip_limit=3, grid=(8, 8), thresh=225):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    dst = clahe.apply(img)
    th = dst.copy()
    th[dst > thresh] = 255
    return th


def encodeDecode(img, ch, quality=5, ext='.jpg'):
    """
    入力された画像を圧縮する
    ※詳細はencodeDecodeNとほぼ同じなので省略
    """

    logger.debug('encodeDecode({},{},{},{})'.format(
        img.shape, ch, quality, ext))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode(ext, img, encode_param)
    if False == result:
        logger.error('image encode failed!')
        logger.error(fileFuncLine())
        exit(1)

    return cv2.imdecode(encimg, getCh(ch))


def encodeDecodeN(imgs, ch, quality=5, ext='.jpg'):
    """
    入力された画像リストを圧縮する
    [in]  imgs:    入力画像リスト
    [in]  ch:      出力画像リストのチャンネル数
    [in]  quality: 圧縮する品質 (1-100)
    [in]  ext:     圧縮する拡張子
    [out] 圧縮画像リスト
    """
    logger.debug('encodeDecodeN(N={})'.format(len(imgs)))
    return [encodeDecode(img, ch, quality, ext) for img in imgs]


def cut(img, size=-1):
    """
    画像を中心から任意のサイズで切り取る
    ※詳細はcutNとほぼ同じなので省略
    """

    logger.debug('cut({},{})'.format(img.shape, size))
    # カットするサイズの半分を計算する
    if size <= 1:
        # サイズが1以下の場合、imgの短辺がカットするサイズになる
        half = np.min(img.shape[:2]) // 2
        logger.debug('\tsize <= 1')
    else:
        half = size // 2

    # 画像の中心位置を計算
    ch, cw = img.shape[0] // 2, img.shape[1] // 2
    return img[ch - half:ch + half, cw - half:cw + half]


def cutN(imgs, size=-1, round_num=-1):
    """
    画像リストの画像を中心から任意のサイズで切り取る
    [in]  img:       カットする画像
    [in]  size:      カットするサイズ（正方形）
    [in]  round_num: 丸める数
    [out] カットされた画像リスト
    """

    logger.debug('cutN(N={})'.format(len(imgs)))
    # 画像のカットを実行
    out_imgs = [cut(img, size) for img in imgs]
    # 切り捨てたい数よりも画像数が少ないと0枚になってしまうので注意
    if(round_num > len(out_imgs)):
        logger.debug('\tround_num > len(out_imgs)')
        round_num = -1

    # バッチサイズの関係などで、画像の数を調整したい時はここで調整する
    # predict.pyなどで分割画像を復元したくなるので縦横の分割数も返す
    if(round_num > 0):
        logger.debug('\tround_num > 0')
        round_len = len(out_imgs) // round_num * round_num
        return np.array(out_imgs[:round_len])
    else:
        return np.array(out_imgs)


def splitSQ(img, size, flg=cv2.BORDER_REPLICATE, w_rate=0.2, array=True):
    """
    入力された画像を正方形に分割する
    ※詳細はsplitSQNとほぼ同じなので省略
    """

    logger.debug('splitSQ({},{},{},{},{})'.format(
        img.shape, size, flg, w_rate, array)
    )

    def arrayChk(x, to_arr):
        logger.debug('arrayChk({},{})'.format(len(x), to_arr))
        # np.array (True)にするか、list (False)にするか選択する
        if to_arr:
            return np.array(x)
        else:
            return x

    # sizeが負数だと分割しないでそのまま返す
    if size <= 1:
        logger.debug('\tsize <= 1')
        return arrayChk(cutN(img), array), (1, 1)

    h, w = img.shape[:2]
    split = (h // size, w // size)

    # sizeが入力画像よりも大きい場合は分割しないでそのまま返す
    if split[0] == 0 or split[1] == 0:
        logger.debug('\tsplit[0] == 0 or split[1] == 0')
        return arrayChk([cut(img)], array), (1, 1)

    # 縦横の分割数を計算する
    if (h / size + w / size) > (h // size + w // size):
        # 画像を分割する際に端が切れてしまうのを防ぐために余白を追加する
        width = int(size * w_rate)
        img = cv2.copyMakeBorder(img, 0, width, 0, width, flg)
        # 画像を分割しやすいように画像サイズを変更する
        h, w = img.shape[:2]
        split = (h // size, w // size)
        img = img[:split[0] * size, :split[1] * size]

    # 画像を分割する
    imgs_2d = [np.vsplit(i, split[0]) for i in np.hsplit(img, split[1])]
    imgs_1d = [x for l in imgs_2d for x in l]
    logger.debug('\tsplit: {}'.format(split))
    return arrayChk(imgs_1d, array), split


def splitSQN(imgs, size, round_num=-1, flg=cv2.BORDER_REPLICATE, w_rate=0.2):
    """
    入力された画像リストを正方形に分割する
    imgsに格納されている画像はサイズが同じであること
    [in]  imgs:      入力画像リスト
    [in]  size:      正方形のサイズ（size x size）
    [in]  round_num: 丸める画像数
    [in]  flg:       境界線のフラグ
    [out] out_imgs:  分割されたnp.array形式の正方形画像リスト
    [out] split:     縦横の分割情報
    """

    logger.debug('splitSQN(N={})'.format(len(imgs)))
    out_imgs = []
    split = []
    for img in imgs:
        i, s = splitSQ(img, size, flg, w_rate, False)
        out_imgs.extend(i)
        split.extend(s)

    # 切り捨てたい数よりも画像数が少ないと0枚になってしまうので注意
    if(round_num > len(out_imgs)):
        logger.debug('\tround_num > len(out_imgs)')
        round_num = -1

    # バッチサイズの関係などで、画像の数を調整したい時はここで調整する
    # predict.pyなどで分割画像を復元したくなるので縦横の分割数も返す
    if(round_num > 0):
        logger.debug('\tround_num > 0')
        round_len = len(out_imgs) // round_num * round_num
        return np.array(out_imgs[:round_len]), (split[0], split[1])
    else:
        return np.array(out_imgs), (split[0], split[1])


def vhstack(imgs, vh_size=None, img_size=None):
    """
    splitSQ(N)された画像リストを連結する
    [in]  imgs:     splitSQ(N)された画像リスト
    [in]  vh_size:  splitSQ(N)時に作成された縦横画像枚数
    [in]  img_size: 本来の画像サイズ
    [out] 連結された元サイズの画像
    """

    if vh_size is None:
        vh_size = (1, len(imgs))

    if len(vh_size) != 2:
        vh_size = (1, len(imgs))

    if vh_size[0] != -1 and vh_size[1] == -1:
        vh_size = (vh_size[0], int(len(imgs)/vh_size[0]+0.5))

    if vh_size[0] == -1 and vh_size[1] != -1:
        vh_size = (int(len(imgs)/vh_size[1]+0.5), vh_size[1])

    if len(imgs) > vh_size[0]*vh_size[1]:
        vh_size = (1, len(imgs))

    if len(imgs) < vh_size[0]*vh_size[1]:
        if len(imgs[0].shape) == 2:
            w, h = imgs[0].shape
            ch = 1
        else:
            w, h, ch = imgs[0].shape

        for i in range(vh_size[0]*vh_size[1] - len(imgs)):
            imgs.append(black(w, h, ch))

    buf = [np.vstack(imgs[i * vh_size[0]: (i + 1) * vh_size[0]])
           for i in range(vh_size[1])]

    if img_size is None:
        return np.hstack(buf)
    else:
        return np.hstack(buf)[: img_size[0], : img_size[1]]


def rotate(img, angle, scale, border=(0, 0, 0)):
    """
    画像を回転（反転）させる
    [in]  img:    回転させる画像
    [in]  angle:  回転させる角度
    [in]  scale:  拡大率
    [in]  border: 回転時の画像情報がない場所を埋める色
    [out] 回転させた画像
    """

    logger.debug('rotate({},{},{},{})'.format(
        img.shape, angle, scale, border)
    )
    size = (img.shape[1], img.shape[0])
    mat = cv2.getRotationMatrix2D((size[0] // 2, size[1] // 2), angle, scale)
    return cv2.warpAffine(img, mat, size, flags=cv2.INTER_CUBIC, borderValue=border)


def rotateR(img, level=[-10, 10], scale=1.2, border=(0, 0, 0)):
    """
    ランダムに画像を回転させる
    ※詳細はrotateRNとほぼ同じなので省略
    """

    logger.debug('rotateR({},{},{},{})'.format(
        img.shape, level, scale, border)
    )
    angle = np.random.randint(level[0], level[1])
    logger.debug('\tangle: {}'.format(angle))
    return rotate(img, angle, scale, border), angle


def rotateRN(imgs, num, level=[-10, 10], scale=1.2, border=(0, 0, 0)):
    """
    画像リストをランダムに画像を回転させる
    [in]  img:   回転させる画像
    [in]  num:   繰り返し数
    [in]  level: 回転させる角度の範囲
    [in]  scale: 拡大率
    [in]  border: 回転時の画像情報がない場所を埋める色
    [out] 回転させた画像リスト
    [out] 回転させた角度リスト
    """

    logger.debug('rotateRN(N={},{})'.format(len(imgs), num))
    out_imgs = []
    out_angle = []
    for n in range(num):
        for img in imgs:
            i, a = rotateR(img, level, scale, border)
            out_imgs.append(i)
            out_angle.append(a)

    return np.array(out_imgs), np.array(out_angle)


def flip(img, num=2):
    """
    画像を回転させてデータ数を水増しする
    ※詳細はflipNとほぼ同じなので省略
    """

    logger.debug('flip({},{})'.format(img.shape, num))
    if num < 1:
        logger.debug('\tnum < 1')
        return [img]

    horizontal = 0
    vertical = 1
    # ベース
    out_imgs = [img.copy()]
    # 上下反転を追加
    f = cv2.flip(img, horizontal)
    out_imgs.append(f)
    if num > 1:
        logger.debug('\tnum > 1')
        # 左右反転を追加
        f = cv2.flip(img, vertical)
        out_imgs.append(f)

    if num > 2:
        logger.debug('\tnum > 2')
        # 上下左右反転を追加
        f = cv2.flip(cv2.flip(img, horizontal), vertical)
        out_imgs.append(f)

    return out_imgs


def flipR(img):
    """
    入力画像をランダムに反転させる
    [in]  反転させたい入力画像
    [out] 反転させた入力画像
    """

    n = np.random.randint(0, 3)
    if n == 2:
        return img
    else:
        return cv2.flip(img, n)


def flipN(imgs, num=2):
    """
    画像を回転させてデータ数を水増しする
    [in]  imgs:     入力画像リスト
    [in]  num:      水増しする数（最大4倍）
    [out] out_imgs: 出力画像リスト
    """

    logger.debug('flipN(N={},{})'.format(len(imgs), num))
    if num < 1:
        logger.debug('\tnum < 1')
        return np.array(imgs)

    horizontal = 0
    vertical = 1
    # ベース
    out_imgs = imgs.copy()
    # 上下反転を追加
    f = [cv2.flip(i, horizontal) for i in imgs]
    out_imgs.extend(f)
    if num > 1:
        logger.debug('\tnum > 1')
        # 左右反転を追加
        f = [cv2.flip(i, vertical) for i in imgs]
        out_imgs.extend(f)

    if num > 2:
        logger.debug('\tnum > 2')
        # 上下左右反転を追加
        f = [cv2.flip(cv2.flip(i, vertical), horizontal) for i in imgs]
        out_imgs.extend(f)

    return np.array(out_imgs)


def resize(img, rate, flg=cv2.INTER_NEAREST):
    """
    画像サイズを変更する
    [in] img:  N倍にする画像
    [in] rate: 倍率
    [in] flg:  N倍にする時のフラグ
    [out] N倍にされた画像リスト
    """

    if rate < 0:
        logger.debug('resize({},{},{})'.format(
            img.shape, rate, flg
        ))
        return img

    size = (int(img.shape[1] * rate),
            int(img.shape[0] * rate))
    logger.debug('resize({},{},{})->{}'.format(
        img.shape, rate, flg, size
    ))
    return cv2.resize(img, size, flg)


def resizeP(img, pixel, flg=cv2.INTER_NEAREST):
    """
    画像サイズを変更する
    [in] img:   サイズを変更する画像
    [in] pixel: 短辺の幅
    [in] flg:   サイズを変更する時のフラグ
    [out] サイズを変更した画像リスト
    """

    logger.debug('resizeP({},{},{})'.format(img.shape, pixel, flg))
    r_img = resize(img, pixel / np.min(img.shape[:2]), flg)
    logger.debug('\tr_img: {}'.format(r_img.shape))
    b_img = cv2.copyMakeBorder(
        r_img, 0, 2, 0, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    logger.debug('\tb_img: {}'.format(b_img.shape))
    return b_img[:pixel, :pixel]


def resizeN(imgs, rate, flg=cv2.INTER_NEAREST):
    """
    画像リストの画像を全てサイズ変更する
    [in] img:  N倍にする画像
    [in] rate: 倍率
    [in] flg:  N倍にする時のフラグ
    [out] N倍にされた画像リスト
    """

    logger.debug('resizeN(N={})'.format(len(imgs)))
    return np.array([resize(img, rate, flg) for img in imgs])


def size2x(imgs, flg=cv2.INTER_NEAREST):
    """
    画像のサイズを2倍にする
    [in] imgs: 2倍にする画像リスト
    [in] flg:  2倍にする時のフラグ
    [out] 2倍にされた画像リスト
    """

    logger.debug('size2x(N={},{})'.format(len(imgs), flg))
    rate = 2
    return [resize(i, rate, flg) for i in imgs]
