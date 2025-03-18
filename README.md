# 吉村（修論：RGB動画版）の引き継ぎコード
VMDデータセットにおけるVMDNetと提案法の比較実験のためのコード．

## Requirement
- Python 3.12.3
- pytorch 2.5.0

## VMD dataset
Linら[1]が作成したVMD データセットは，洗面台や家具などの平面鏡を動画撮影したデータセットで，フレームごとに
鏡面領域の正解マスク画像が付与されている．データセットの内容は，
学習用（```train_origin```）に143 本（7,835 フレーム），テスト用（```test```）に126 本
（7,152）が用意されている．[Google Drive link](https://drive.google.com/drive/folders/1ECfkY8RyAyjYu9lTm7vvvU6ZE2Tg2Ush?usp=drive_link)


## Data structure
- Google driveにアップロードされたzipファイル（```train_origin.zip```,```test.zip```）を```./VMD```ダウンロードし全て展開する
- 学習データ```train_origin```をモデルの学習用と検証用に5foldに分割するために，```VMD/split_train_val_fold.py```を実行する．

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
実験環境を、以下の手順に沿って構築
1. リポジトリーのclone
```bash
$ git clone https://github.com/yossi-yuto/Yoshimura_master_2024_VMD.git
```
2.  ```requirements.txt``` に記載されたパーケージのインストール
```bash
$ pip install -r requirements.txt
```
3. ```./scripts```ディレクトリに移動
```bash
cd scripts
```

## How to run
提案法のモデルを5foldの交差検証で実施する場合，以下のように実行．
```bash 
/scripts$ source pipeline_proposed_fold.sh {GPU_NUM} {date}
```
`{GPU_NUM}`はGPUのデバイスを指定し，`{date}`は実行日時を記載．

### Example
GPUデバイスの０番を使用し、2025年2月10日に実行する場合、以下のコマンドを実行.

```bash
/scripts$ source pipeline_proposed_fold.sh 0 20250210
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


### Reference

[1]Jiaying Lin, Xin Tan, Rynson W.H. Lau, "Learning To Detect Mirrors From Videos via Dual Correspondences," Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9109–9118, June 2023. [project page](https://cvpr.thecvf.com/virtual/2023/poster/21597)

