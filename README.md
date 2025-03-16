# 吉村（修論）の引き継ぎコード
VMDデータセットにおけるVMDNetと提案法の比較実験のためのコード．

## PythonとPytorchのバージョン
- Python 3.12.3
- pytorch 2.5.0



## VMDデータセット
VMD データセットは，洗面台や家具などの平面鏡を動画撮影したデータセットで，フレームごとに
鏡面領域の正解マスク画像が付与されている．学習用に143 本（7,835 フレーム），テスト用に126 本
（7,152）が用意されている．
- [Google Drive](https://drive.google.com/drive/folders/1ECfkY8RyAyjYu9lTm7vvvU6ZE2Tg2Ush?usp=drive_link)

また，学習データセット```train_origin```のうち5foldのデータセットの分割をする際には、```VMD/split_train_val_fold.py```を実行

## データ構造
- Google driveにアップロードされたzipファイル（```train_origin.zip```,```test.zip```）を```./VMD```ダウンロードし展開
- 

```
./VMD
├── train_origin
│   ├── 113_27 # ビデオID
│   │   ├── JPEGImages #フレーム単位でRGB画像が格納
│   │   │   ├── 0001.jpg
│   │   │   ├── 0002.jpg
│   │   │   ├── 0003.jpg
│   │   │   ├── 0004.jpg
│   │   │   └── ...
│   │   └── SegmentationClassPNG #フレーム単位で鏡面領域のマスク画像が格納
│   │       ├── 0001.png
│   │       ├── 0002.png
│   │       ├── 0003.png
│   │       ├── 0004.png
│   │       └── ...
│   ├── 113_36
│
└── test
    ├── 000_0
    │   ├── JPEGImages #フレーム単位でRGB画像が格納
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   ├── 0003.jpg
    │   │   ├── 0004.jpg
    │   │   └── ...
    │   └── SegmentationClassPNG #フレーム単位で鏡面領域のマスク画像が格納
    │       ├── 0001.png
    │       ├── 0002.png
    │       ├── 0003.png
    │       ├── 0004.png
    │       └── ...
    ├── 000_1

```

## Installation
実験は以下の手順により再現
1. リポジトリーのclone
```bash
$ git clone https://github.com/yossi-yuto/Yoshimura_master_2024_VMD.git
```
2.  ```requirements.txt``` に記載されたパーケージのインストール
```bash
$ pip install -r requirements.txt
```

## 実行方法
提案法のモデルを5foldの交差検証で実施する場合，以下のように実行．
```bash 
$ source pipeline_proposed_fols.sh {GPU_NUM} {date}
```
`{GPU_NUM}`はGPUのデバイスを指定し，`{date}`は実行日時を記載．

### 実行例
GPUデバイスの０番を使用し、2025年2月10日に実行する場合、以下のコマンドを実行.

```bash
$ source pipeline_spherical.sh 0 20250210
```
なお、実験結果は```./scripts/experiment_results```に以下のように5fold分が作成される
```
./scripts
└── experiment_results
    ├── 20250210_fold_0
    ├── 20250210_fold_1
    ├── ...
    └── 20250210_fold_4
```


