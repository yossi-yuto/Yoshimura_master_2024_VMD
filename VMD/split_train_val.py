import os
import shutil
import random

# Source directory
source_dir = "train_origin"

# Destination directories
train_dir = "./train"
val_dir = "./val"

# Get a list of all folders in the source directory
folders = [i for i in os.listdir(source_dir) if os.path.splitext(i)[1] != ".txt"]

# Calculate the number of folders to copy for train and val
num_train = int(len(folders) * 0.8)
num_val = len(folders) - num_train

# Shuffle the list of folders
random.shuffle(folders)

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
