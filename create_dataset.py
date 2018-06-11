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
    parser.add_argument('-fn', '--font_num', type=int, default=20,
                        help='フォント数 [default: 20]')
    parser.add_argument('-in', '--img_num', type=int, default=1000,
                        help='画像生成数 [default: 1000]')
    parser.add_argument('-t', '--train_per_all', type=float, default=0.9,
                        help='画像数に対する学習用画像の割合 [default: 0.9]')
    parser.add_argument('-o', '--out_path', default='./dataset/',
                        help='データセット保存先 (default: ./dataset/)')
    args = parser.parse_args()
    F.argsPrint(args)
    return args


def rndChoice(imgs, num):
    return imgs[np.random.choice(range(imgs.shape[0]), num, replace=False)]


def create(pre_fonts, img_size, font_num, img_num, img_buf=20):
    y = []
    size = (img_buf // 2, img_size + img_buf // 2)
    # 使用するフォントをランダムで事前に選択しておく
    # np.random.choiceはかなりコストの大きい処理なので注意
    all_fonts = [rndChoice(pre_fonts, font_num) for i in range(img_num)]
    for fonts in all_fonts:
        # フォントをセットする背景の生成
        img = IMG.white(img_size + img_buf, img_size + img_buf, 3)
        for font in fonts:
            # 上辺と左辺の枠を消してフォントを一つ選択する
            img, _ = IMG.paste(font[1:, 1:, ], img, mask_flg=False)

        # 画像の端を除去し、グレースケールに変換
        y.append(img[size[0]:size[1], size[0]:size[1]])

    return y


def getPath(out_path, i, zfill=6, str_len=12):
    folder = os.path.join(out_path, str(i).zfill(zfill))
    name = GET.datetimeSHA(GET.randomStr(10), str_len=str_len)
    return F.getFilePath(folder, name, '.jpg')


def main(args):

    # フォント画像の読み込み
    print('read images...')
    fonts = IMG.readN(args.font, 3)

    # フォント画像をフォントごとに分割する
    print('split images...')
    fonts, _ = IMG.splitSQN(fonts, args.font_size)

    # 正解画像の生成と保存
    max_folder = 4000
    print('create and save images...')
    for i in range(0, args.img_num, max_folder):
        num = np.min([max_folder, args.img_num])
        [cv2.imwrite(getPath(args.out_path, i), j)
         for j in create(fonts, args.img_size, args.font_num, num)]

    print('save param...')
    F.dict2json(args.out_path, 'dataset', F.args2dict(args))


if __name__ == '__main__':
    st = time.time()
    main(command())
    print('{0:8.3f} [s]'.format(time.time()-st))
