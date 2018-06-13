#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像を読み込んでデータセットを作成する'
#

import logging
# basicConfig()は、 debug()やinfo()を最初に呼び出す"前"に呼び出すこと
level = logging.INFO
logging.basicConfig(format='%(message)s')
logging.getLogger('Tools').setLevel(level=level)

import os
import cv2
import time
import argparse
import numpy as np

from multiprocessing import Pool

import Tools.imgfunc as IMG
import Tools.getfunc as GET
import Tools.func as F


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('font',  nargs='+', help='使用する入力フォント画像')
    parser.add_argument('-is', '--img_size', type=int, default=128,
                        help='生成される画像サイズ [default: 128 pixel]')
    parser.add_argument('-fs', '--font_size', type=int, default=32,
                        help='使用するフォントのサイズ [default: 32x32]')
    parser.add_argument('-fn', '--font_num', type=int, default=16,
                        help='フォント数 [default: 16]')
    parser.add_argument('-in', '--img_num', type=int, default=1000,
                        help='画像生成数 [default: 1000]')
    parser.add_argument('-o', '--out_path', default='./dataset/',
                        help='データセット保存先 (default: ./dataset/)')
    args = parser.parse_args()
    F.argsPrint(args)
    return args


def create(fonts, img, size):
    for font in fonts:
        # 上辺と左辺の枠を消してフォントを一つ選択する
        img, _ = IMG.paste(font[1:, 1:, ], img, mask_flg=False)

    return img[size[0]:size[1], size[0]:size[1]]


def w_create(args):
    return create(*args)


def createN(pre_fonts, img_size, font_num, img_num, img_buf=20, processes=4):
    #####
    # フォント画像と背景画像の準備
    #####

    # 背景画像を生成する
    img = IMG.white(img_size + img_buf, img_size + img_buf, 3)
    # img_bufで拡張した背景画像をimg_sizeの幅で切り取るために使用する
    size = (img_buf // 2, img_size + img_buf // 2)
    # 使用するフォントをランダムシャッフルしておく
    fonts = np.array(pre_fonts)
    fonts = fonts[np.random.permutation(range(len(pre_fonts)))]

    #####
    # ここからマルチプロセス処理
    #####

    # ラッパー関数用に引数をリストで格納する
    param = [(fonts[i:i+font_num], img, size)
             for i in range(0, len(fonts), font_num)][:img_num]
    # 使用するプロセス数を指定し、実行し、クローズする
    p = Pool(processes=processes)
    #out = p.map(w_create, param)
    out = p.imap_unordered(w_create, param)
    p.close()
    return out


def getPath(out_path, i, zfill=4, str_len=12):
    folder = os.path.join(out_path, str(i).zfill(zfill))
    name = GET.datetimeSHA(GET.randomStr(10), str_len=str_len)
    return F.getFilePath(folder, name, '.jpg')


def main(args):
    # フォント画像の読み込みと分割する
    print('read and split images...')
    fonts, _ = IMG.splitSQN(IMG.readN(args.font, 3), args.font_size)

    print(fonts.shape)
    param = F.args2dict(args)
    param['shape'] = fonts.shape

    # 正解画像の生成と保存
    max_folder = 4000
    proc = 7
    print('create and save images...')
    for i in range(0, args.img_num, max_folder):
        num = np.min([max_folder, args.img_num-i])
        [cv2.imwrite(getPath(args.out_path, i//max_folder), j)
         for j in createN(fonts, args.img_size, args.font_num, num, processes=proc)]

    print('save param...')
    F.dict2json(args.out_path, 'dataset', param)


if __name__ == '__main__':
    st = time.time()
    main(command())
    print('{0:8.3f} [s]'.format(time.time()-st))
