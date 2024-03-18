import os
import shutil
import random

def split_dataset(original_dir, segment_dir, train_dir, val_dir, split_ratio):
    # 元の画像ファイルを取得
    original_files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]

    # ランダムにシャッフル
    random.shuffle(original_files)

    # トレーニングセットとバリデーションセットのサイズを計算
    total_images = len(original_files)
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
    for i, original_file in enumerate(original_files):
        original_path = os.path.join(original_dir, original_file)
        segment_path = os.path.join(segment_dir, f"{os.path.splitext(os.path.basename(original_file))[0]}.png")  # セグメンテーション画像のパス
        segment_file = f"{os.path.splitext(os.path.basename(original_file))[0]}.png"
        #print(f"{segment_directory}/{os.path.split(os.path.basename(original_file))[0]}.png")
        #print(os.path.exists(f"{segment_directory}/{os.path.splitext(os.path.basename(original_file))[0]}.png"))
        if os.path.exists(segment_path):
            if i < train_size:
                target_original_path = os.path.join(f"{train_dir}/original", original_file)
                target_segment_path = os.path.join(f"{train_dir}/segment", segment_file)
            else:
                target_original_path = os.path.join(f"{val_dir}/original", original_file)
                target_segment_path = os.path.join(f"{val_dir}/segment", segment_file)
            shutil.copyfile(original_path, target_original_path)
            shutil.copyfile(segment_path, target_segment_path)

if __name__ == "__main__":
    original_directory = "dataset/original"
    segment_directory = "dataset/segment"
    train_directory = "dataset/train"
    val_directory = "dataset/val"
    split_ratio = 0.8  # トレーニングセットの割合

    split_dataset(original_directory, segment_directory, train_directory, val_directory, split_ratio)
