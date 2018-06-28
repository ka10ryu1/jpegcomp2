#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '学習メイン部'
#

import cv2
import os
import logging
# basicConfig()は、 debug()やinfo()を最初に呼び出す"前"に呼び出すこと
level = logging.INFO
logging.basicConfig(format='%(message)s')
logging.getLogger('Tools').setLevel(level=level)

import argparse
import numpy as np

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
# from chainer.datasets import LabeledImageDataset

from Lib.plot_report_log import PlotReportLog
from Lib.network import JC_DDUU as JC
from Lib.read_dataset_CV2 import LabeledImageDataset
import Tools.imgfunc as I
import Tools.getfunc as G
import Tools.func as F
import Tools.pruning as pruning


class ResizeAndEncdecImgDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, rate, quality=5, dtype=np.float32):
        self._dataset = dataset
        self._rate = rate
        self._quality = quality
        self._dtype = dtype
        self._len = len(self._dataset)

    def __len__(self):
        # データセットの数を返します
        return self._len

    def get_example(self, i):
        # データセットのインデックスを受け取って、データを返します
        inputs = self._dataset[i]
        z, _ = inputs
        z = z.transpose(1, 2, 0).astype(np.uint8)
        z = cv2.cvtColor(z, cv2.COLOR_RGB2GRAY)
        x = I.arr.img2arr(I.cnv.encodeDecode(z, I.io.getCh(1), self._quality))
        y = I.arr.img2arr(I.cnv.resize(z, self._rate))
        return x.astype(self._dtype), y.astype(self._dtype)


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('-i', '--in_path', default='./result/',
                        help='入力データセットのフォルダ [default: ./result/]')
    parser.add_argument('-u', '--unit', type=int, default=2, metavar='INT',
                        help='ネットワークのユニット数 [default: 2]')
    parser.add_argument('-sr', '--shuffle_rate', type=int, default=2, metavar='INT_VAL',
                        help='PSの拡大率 [default: 2]')
    parser.add_argument('-a1', '--actfun1', default='relu',
                        choices=('relu', 'elu', 'c_relu', 'l_relu',
                                 'sigmoid', 'h_sigmoid', 'tanh', 's_plus'),
                        help='活性化関数(1) [default: relu]')
    parser.add_argument('-a2', '--actfun2', default='sigmoid',
                        choices=('sigmoid', 'relu', 'elu', 'c_relu',
                                 'l_relu', 'h_sigmoid', 'tanh', 's_plus'),
                        help='活性化関数(2) [default: sigmoid]')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, metavar='FLOAT',
                        help='ドロップアウト率（0〜0.9、0で不使用）[default: 0.2]')
    parser.add_argument('-opt', '--optimizer', default='adam',
                        choices=('adam', 'ada_d', 'ada_g', 'm_sgd',
                                 'n_ag', 'rmsp', 'rmsp_g', 'sgd', 'smorms'),
                        help='オプティマイザ [default: adam]')
    parser.add_argument('-lf', '--lossfun', default='mse',
                        choices=('mse', 'mae', 'ber', 'gauss_kl'),
                        help='損失関数 [default: mse]')
    parser.add_argument('-p', '--pruning', type=float, default=0.33, metavar='FLOAT',
                        help='pruning率（snapshot使用時のみ効果あり） [default: 0.5]')
    parser.add_argument('-b', '--batchsize', type=int, default=100, metavar='INT',
                        help='ミニバッチサイズ [default: 100]')
    parser.add_argument('-e', '--epoch', type=int, default=10, metavar='INT',
                        help='学習のエポック数 [default 10]')
    parser.add_argument('-f', '--frequency', type=int, default=-1, metavar='INT',
                        help='スナップショット周期 [default: -1]')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1, metavar='INT',
                        help='使用するGPUのID [default -1]')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    parser.add_argument('-r', '--resume', default='',
                        help='使用するスナップショットのパス[default: no use]')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='学習過程をPNG形式で出力しない場合に使用する')
    parser.add_argument('--only_check', action='store_true',
                        help='オプション引数が正しく設定されているかチェックする')
    args = parser.parse_args()
    F.argsPrint(args)
    return args


def getDataset(folder, shuffle_rate):
    # 探索するフォルダがなければ終了
    if not os.path.isdir(folder):
        print('[Error] folder not found:', folder)
        print(F.fileFuncLine())
        exit()

    # 学習用データとテスト用データを発見したらTrueにする
    train_flg = False
    test_flg = False
    n_out = 0
    for l in os.listdir(folder):
        name, ext = os.path.splitext(os.path.basename(l))
        if os.path.isdir(l):
            pass
        elif('train_' in name)and('.txt' in ext)and(train_flg is False):
            train = LabeledImageDataset(os.path.join(folder, l))
            train = ResizeAndEncdecImgDataset(train, shuffle_rate)
            train_flg = True
            n_out = int(name.split('_')[1])
        elif('test_' in name)and('.txt' in ext)and(test_flg is False):
            test = LabeledImageDataset(os.path.join(folder, l))
            test = ResizeAndEncdecImgDataset(test, shuffle_rate)
            test_flg = True
            n_out = int(name.split('_')[1])

    return train, test, n_out


def main(args):
    # 各種データをユニークな名前で保存するために時刻情報を取得する
    exec_time = G.datetimeSHA()

    # 活性化関数を取得する
    actfun1 = G.actfun(args.actfun1)
    actfun2 = G.actfun(args.actfun2)
    model = L.Classifier(
        JC(n_unit=args.unit, rate=args.shuffle_rate,
           actfun1=actfun1, actfun2=actfun2, dropout=args.dropout,
           view=args.only_check),
        lossfun=G.lossfun(args.lossfun)
    )
    # Accuracyは今回使用しないのでFalseにする
    # もしも使用したいのであれば、自分でAccuracyを評価する関数を作成する必要あり？
    model.compute_accuracy = False

    # Setup an optimizer
    optimizer = G.optimizer(args.optimizer).setup(model)

    # Load dataset
    train, test, _ = getDataset(args.in_path, args.shuffle_rate)
    # predict.pyでモデルを決定する際に必要なので記憶しておく
    model_param = F.args2dict(args)
    model_param['shape'] = train[0][0].shape

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu_id
    )
    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out_path
    )

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu_id))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(
        extensions.dump_graph('main/loss', out_name=exec_time + '_graph.dot')
    )

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(
        extensions.snapshot(filename=exec_time + '_{.updater.epoch}.snapshot'),
        trigger=(frequency, 'epoch')
    )

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(log_name=exec_time + '.log'))
    # trainer.extend(extensions.observe_lr())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            PlotReportLog(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='loss.png')
        )

        # trainer.extend(
        #     PlotReportLog(['lr'],
        #                   'epoch', file_name='lr.png', val_pos=(-80, -60))
        # )

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport([
        'epoch',
        'main/loss',
        'validation/main/loss',
        # 'lr',
        'elapsed_time'
    ]))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
        # Set pruning
        # http://tosaka2.hatenablog.com/entry/2017/11/17/194051
        masks = pruning.create_model_mask(model, args.pruning, args.gpu_id)
        trainer.extend(pruning.pruned(model, masks))

    # Make a specified GPU current
    if args.gpu_id >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu_id).use()
        # Copy the model to the GPU
        model.to_gpu()
        chainer.global_config.autotune = True
    else:
        model.to_intel64()

    # predict.pyでモデルのパラメータを読み込むjson形式で保存する
    if args.only_check is False:
        F.dict2json(args.out_path, exec_time + '_train', model_param)

    # Run the training
    trainer.run()

    # 最後にモデルを保存する
    # スナップショットを使ってもいいが、
    # スナップショットはファイルサイズが大きい
    chainer.serializers.save_npz(
        F.getFilePath(args.out_path, exec_time, '.model'),
        model
    )


if __name__ == '__main__':
    main(command())
