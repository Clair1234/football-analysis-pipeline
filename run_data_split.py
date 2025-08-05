import os
import random
import shutil

# --- Config ---
source_folder = "./data/processed"
train_folder = "./data/processed/training"
test_folder = "./data/processed/testing"
train_ratio = 0.8  # 80% for training, 20% for testing
seed = 42  # for reproducibility

# --- Prepare ---
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get all file paths
all_files = [
    f for f in os.listdir(source_folder)
    if os.path.isfile(os.path.join(source_folder, f)) and f.lower().endswith(".parquet")
]
random.seed(seed)
random.shuffle(all_files)

# Split
split_idx = int(train_ratio * len(all_files))
train_files = all_files[:split_idx]
test_files = all_files[split_idx:]

# Move files
for f in train_files:
    shutil.move(os.path.join(source_folder, f), os.path.join(train_folder, f))

for f in test_files:
    shutil.move(os.path.join(source_folder, f), os.path.join(test_folder, f))

print(f"Moved {len(train_files)} files to '{train_folder}' and {len(test_files)} files to '{test_folder}'")
