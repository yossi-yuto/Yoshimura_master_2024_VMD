import os
import shutil

def split_list(lst, n):
    """
    リストを n 分割する関数。

    Args:
        lst (list): 分割したいリスト。
        n (int): 分割数。

    Returns:
        list: n 分割されたリストのリスト。
    """
    k, m = divmod(len(lst), n)  # k: 各部分の基本サイズ, m: 余り
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


# パラメータ設定
fold_num = 5
source_dir = "train_origin"

# ソースディレクトリ内のフォルダを取得
video_folders = sorted([i for i in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, i))])

# フォルダを5分割
fold_val_list = split_list(video_folders, fold_num)

# ディレクトリ分割とコピー
for i in range(fold_num):
    train_dir = f"train_fold_{i}"
    val_dir = f"val_fold_{i}"

    # 出力ディレクトリの作成（存在する場合は削除して再作成）
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)

    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)

    # 検証用フォルダ（fold_val_list[i]）を val にコピー
    for folder in fold_val_list[i]:
        shutil.copytree(os.path.join(source_dir, folder), os.path.join(val_dir, folder))

    # 残りのフォルダを train にコピー
    train_folders = [folder for j, fold in enumerate(fold_val_list) if j != i for folder in fold]
    for folder in train_folders:
        shutil.copytree(os.path.join(source_dir, folder), os.path.join(train_dir, folder))

print("フォルダ分割とコピーが完了しました！")
