#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'データセットテキスト生成部分'
#

import os
import time
import argparse
import numpy as np
from glob import glob

import Tools.func as F
import Tools.imgfunc as IMG


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('-i', '--img_root_path', default='dataset/',
                        help='テキストデータを作成したいデータセットのルートパス')
    parser.add_argument('-t', '--train_per_all', type=float, default=0.9,
                        help='画像数に対する学習用画像の割合 [default: 0.9]')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='データの保存先 (default: ./result/)')
    args = parser.parse_args()
    F.argsPrint(args)
    return args


def writeTXT(path, data):
    """
    テキストファイルを書き出す
    [in] folder: テキストを保存するフォルダ
    [in] name:   テキストの名前
    [in] data:   保存するデータ

    dataは[(path1, val1), (path2, val2), ... , (pathN, valN)]の形式であること
    pathN: N番目の画像パス
    valN:  N番目の画像の分類番号
    """

    with open(path, 'w') as f:
        [f.write('./' + i + ' ' + j + '\n') for i, j in data]


def str2int(in_str):
    """
    入力された数値がintに変換できる場合はintを返す
    intで返せない場合はエラーを出力する
    """

    val = 0
    try:
        val = int(in_str)
    except:
        print('ERROR:', in_str)
        val = -1

    return val


def main(args):
    # 画像データを探索し、画像データのパスと、サブディレクトリの値を格納する
    search = glob(os.path.join(args.img_root_path, '**'), recursive=True)
    data = [(img, str2int(img.split('/')[1])) for img in search
            if IMG.isImgPath(img, True)]
    # ラベルの数を数える
    label_num = len(np.unique(np.array([i for _, i in data])))
    print('label num: ', label_num)
    # 取得したデータをランダムに学習用とテスト用に分類する
    data_arr = np.array(data)
    data_len = len(data_arr)
    shuffle = np.random.permutation(range(data_len))
    train_size = int(data_len * args.train_per_all)
    data = {'train': data_arr[shuffle[:train_size]],
            'test': data_arr[shuffle[train_size:]]}

    for key, val in data.items():
        print(key, val.shape)
        print(val)

        # chainer.datasets.LabeledImageDataset形式で出力する
        path = F.getFilePath(args.out_path, key + '_' +
                             str(label_num).zfill(3), '.txt')
        writeTXT(path, val)


if __name__ == '__main__':
    st = time.time()
    main(command())
    print('{0:8.3f} [s]'.format(time.time()-st))
