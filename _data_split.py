import os
import shutil
import random

def split_dataset(original_dir, segment_dir, train_dir, val_dir, split_ratio):
    # 共通するファイル名を取得
    common_files = set(os.listdir(original_dir)) & set(os.listdir(segment_dir))

    # ランダムにシャッフル
    common_files = list(common_files)
    random.shuffle(common_files)

    # トレーニングセットとバリデーションセットのサイズを計算
    total_images = len(common_files)
    train_size = int(total_images * split_ratio)
    val_size = total_images - train_size

    # train ディレクトリが存在しない場合は作成
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    # train/original ディレクトリが存在しない場合は作成
    if not os.path.exists(f"{train_dir}/original"):
        os.makedirs(f"{train_dir}/original")
    # train/original ディレクトリが存在しない場合は作成
    if not os.path.exists(f"{train_dir}/segment"):
        os.makedirs(f"{train_dir}/segment")

    # val ディレクトリが存在しない場合は作成
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    # val/original ディレクトリが存在しない場合は作成
    if not os.path.exists(f"{val_dir}/original"):
        os.makedirs(f"{val_dir}/original")
    # val/original ディレクトリが存在しない場合は作成
    if not os.path.exists(f"{val_dir}/segment"):
        os.makedirs(f"{val_dir}/segment")
        
    # 画像をコピー
    for i, filename in enumerate(common_files):
        original_path = os.path.join(original_dir, filename)
        segment_path = os.path.join(segment_dir, filename)
        if i < train_size:
            target_original_path = os.path.join(f"{train_dir}/original", filename)
            print(target_original_path)
            target_segment_path = os.path.join(f"{train_dir}/segment", filename)
        else:
            target_original_path = os.path.join(f"{val_dir}/original", filename)
            target_segment_path = os.path.join(f"{val_dir}/segment", filename)
        shutil.copyfile(original_path, target_original_path)
        shutil.copyfile(segment_path, target_segment_path)

if __name__ == "__main__":
    original_directory = "dataset/original"
    segment_directory = "dataset/segment"
    train_directory = "dataset/train"
    val_directory = "dataset/val"
    split_ratio = 0.8  # トレーニングセットの割合

    split_dataset(original_directory, segment_directory, train_directory, val_directory, split_ratio)
