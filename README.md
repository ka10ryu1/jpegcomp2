# 概要

圧縮されたJpeg画像の復元

## 学習結果その1（5%に圧縮した画像を復元）

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/concat-100.jpg" width="640px">

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/concat-110.jpg" width="640px">

## 学習結果その2（20%に圧縮した画像を復元）

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/concat-000.jpg" width="640px">

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/concat-010.jpg" width="640px">

- **Original** : 圧縮なしの画像
- **Compression** : Originalを圧縮した画像
- **Restoration** : Compressionを学習によって復元した画像

# 動作環境

## $ cat /etc/issue

- **Ubuntu** 16.04.4 LTS

## $ Tools/version_check.py

- **Numpy** 1.14.2
- **OpenCV** 3.4.0
- **iDeep** None
- **Cupy** 4.0.0
- **Python** 3.5
- **Matplotlib** 2.2.2
- **Chainer** 4.0.0
- **Pillow** 5.1.0

# ファイル構成

## 生成方法

```console
$ ls `find ./ -maxdepth 3 -type f -print` | xargs grep 'help = ' --include=*.py >& log.txt
$ tree >& log.txt
```

## ファイル

```console
.
├── FontData
│   ├── 00_Arial_Unicode_MS.png
│   ├── 01_HGPｺﾞｼｯｸE.png
│   ├── 03_HGPｺﾞｼｯｸM.png
│   ├── 04_HGP教科書体.png
│   ├── 05_HGP行書体.png
│   ├── 06_HGP創英ﾌﾟﾚｾﾞﾝｽEB.png
│   ├── 07_HGP創英角ｺﾞｼｯｸUB.png
│   ├── 08_HGP創英角ﾎﾟｯﾌﾟ体.png
│   ├── 09_HGP明朝B.png
│   ├── 10_HGP明朝E.png
│   ├── 11_HG丸ｺﾞｼｯｸM-PRO.png
│   ├── 12_Meiryo_UI.png
│   ├── 13_游ゴシック.png
│   ├── 14_游明朝.png
│   ├── The_Night_of_the_Milky_Way_Train_ch2.PNG > predict用画像
│   └── The_Nighthawk_Star_op.PNG                > predict用画像
├── LICENSE
├── Lib
│   ├── concat_3_images.py > 3枚の画像を連結する（org, comp, restration）
│   ├── network.py         > jpegcompのネットワーク部分
│   ├── plot_report_log.py
│   └── read_dataset_CV2.py
├── Model
│   ├── demo.model
│   ├── param.json
│   └── result.log
├── README.md
├── Tools
├── auto_train.sh
├── clean_all.sh
├── create_dataset.py        > 画像を読み込んでデータセットを作成する
├── predict.py               > モデルとモデルパラメータを利用して推論実行する
├── predict_some_snapshot.py > 複数のsnapshotoとひとつのモデルパラメータを利用してsnapshotの推移を可視化する
├── train.py                 > 学習メイン部
└── write_dataset.py         > データセットテキスト生成部分
 ```

# チュートリアル

## 学習済みモデルを利用する

以下を実行すれば学習済みモデルを利用できるが、CPUだと数分かかるので`-g GPU_ID`推奨。

```console
$ ./predict.py Model/*model Model/*json FontData/The_Night*
```

## 自分で学習データを作成し、学習をしてみる

### 1. データセットの作成

以下を入力して1000枚のデータセットを`dataset`フォルダに生成させる

```console
$ ./create_dataset.py FontData/*.png
```

以下を入力して生成されたデータセットのパスをテキストに出力する

```console
$ ./write_dataset.py
```

以下を入力してテキストが正常に出力されたか確認する。今回は生成した画像1000枚のうち、1800枚を学習用、200枚をテスト用としている。

```console
$ wc result/t*
  100   200  3400 result/test_001.txt
  900  1800 30600 result/train_001.txt
 1000  2000 34000 合計
```

### 2. 学習の実行

以下を入力して学習を開始する。10分程度で学習が終わるはず。

```console
$ ./train.py
```

学習完了時に、`result`フォルダに以下が生成されていればOK。先頭の文字列は日付と時間から算出された値であるため実行ごとに異なる。

- `*.log`
- `*.model`
- `*_10.snapshot`
- `*_graph.dot`
- `*_train.json`
- `loss.png`

### 3. 学習で作成されたモデルを使用する

以下を実行して学習済みモデルを推論実行してみる

```console
$ ./predict.py result/*model result/*json FontData/The_Night*
```

推論実行後に、`result`フォルダに`comp-*.jpg`と`concat-*.jpg`が生成されていればOK。

# その他の機能

## 生成されたデータの削除

```console
$ ./clean_all.sh
```

## ハイパーパラメータを変更して自動実行

デフォルトではバッチサイズだけを変更した学習を複数回繰り返す。`-c`オプションを付けることでネットワークの中間層の数などを可視化もできる。


```console
$ ./auto_train.sh
```

※自動実行は現在テストしていません

## スナップショットの進捗具合を可視化する

各スナップショットで推論実行したものを比較することで学習がどれだけ進んでいるかを可視化する。以下でエポックごとにスナップショットを保存してくれる。


```console
$ ./train.py -f 1
```

以下を実行することでスナップショットの遷移を可視化できる。

```console
$ ./predict_some_snapshot.py ./result/ ./FontData/The_Night*
```

以下のような画像が生成される。一番左が正解画像で、右に行くほど新しいスナップショットの結果になる。

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/snapshots.jpg" width="320px">

## ネットワーク層の確認

`train.py`実行時に`--only_check`フラグを入れるとネットワーク層の確認ができる。

```console
$ ./train.py --only_check
```

そうすると以下のようにブロック名、実行時間（累計）、ネットワーク層の数が表示される。ネットワーク層のパラメータを修正した時などに利用すると便利。

```console
:
省略
:
[Network info] JC_DDUU
  Unit:	2
  Out:	1
  Drop out:	0.2
  Act Func:	relu, sigmoid
 0: DownSampleBlock	0.000 s	(100, 1, 128, 128)
 1: DownSampleBlock	0.336 s	(100, 2, 64, 64)
 2: DownSampleBlock	0.413 s	(100, 4, 32, 32)
 3: DownSampleBlock	0.448 s	(100, 8, 16, 16)
 4: DownSampleBlock	0.463 s	(100, 16, 8, 8)
 5: UpSampleBlock	0.475 s	(100, 32, 8, 8)
 6: UpSampleBlock	0.520 s	(100, 10, 16, 16)
 7: UpSampleBlock	0.591 s	(100, 6, 32, 32)
 8: UpSampleBlock	0.789 s	(100, 4, 64, 64)
 9: UpSampleBlock	1.306 s	(100, 2, 128, 128)
Output 2.522 s: (100, 1, 256, 256)
```


## 完全版の実行

以下はGPU必須で、GTX1060でも数日かかる計算量になるので注意。以下の工程で生成された学習モデルとパラメータが`Model`に格納されている。
