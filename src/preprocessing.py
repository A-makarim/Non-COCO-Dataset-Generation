"""
This module contains preprocessing functions for the Mars Probe Detection project.

remove_empty_labels: bool

split_data: 
"""


from pathlib import Path
import random
import shutil


def remove_empty_labels(data_dir: str):
    """
    Only use this if the probe was constantly in view of the camera.
    Remove label files that are empty and their corresponding image files.
    """
    data_path = Path(data_dir)
    labels_dir = data_path / 'labels'
    images_dir = data_path / 'images'

    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            print(f"Removing empty label file: {label_file}")
            image_file = images_dir / f"{label_file.stem}.jpg"  # assuming .jpg images
            if image_file.exists():
                print(f"Removing corresponding image file: {image_file}")
                image_file.unlink() 
            label_file.unlink()

def split_dataset(data_dir: str, train_ratio=0.7, val_ratio=0.2, seed=42):

    """
    Split dataset into train, validation, and test sets.
    Yolo needs images and labels in separate folders for each split.
    """
    random.seed(seed)

    data_path = Path(data_dir)
    images = sorted((data_path / "images").glob("*.jpg"))
    labels = {p.stem: p for p in (data_path / "labels").glob("*.txt")}

    pairs = [(img, labels.get(img.stem)) for img in images]
    pairs = [p for p in pairs if p[1] is not None]

    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:]
    }

    for split, items in splits.items():
        for img, lbl in items:
            (data_path / split / "images").mkdir(parents=True, exist_ok=True)
            (data_path / split / "labels").mkdir(parents=True, exist_ok=True)

            shutil.copy(img, data_path / split / "images" / img.name)
            shutil.copy(lbl, data_path / split / "labels" / lbl.name)


