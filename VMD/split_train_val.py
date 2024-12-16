import os
import shutil
import random

# 固定するシード値（任意の整数を指定）
SEED = 42

# Source directory
source_dir = "train_origin"

# Destination directories
train_dir = "./train"
val_dir = "./val"

# Ensure destination directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get a list of all folders in the source directory
folders = [i for i in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, i))]

# Set random seed for reproducibility
random.seed(SEED)

# Shuffle the list of folders
random.shuffle(folders)

# Calculate the number of folders to copy for train and val
num_train = int(len(folders) * 0.8)
num_val = len(folders) - num_train

# Copy folders to train directory
for folder in folders[:num_train]:
    folder_path = os.path.join(source_dir, folder)
    dest_path = os.path.join(train_dir, folder)
    shutil.copytree(folder_path, dest_path)

# Copy folders to val directory
for folder in folders[num_train:]:
    folder_path = os.path.join(source_dir, folder)
    dest_path = os.path.join(val_dir, folder)
    shutil.copytree(folder_path, dest_path)

print(f"Copied {num_train} folders to {train_dir}")
print(f"Copied {num_val} folders to {val_dir}")
