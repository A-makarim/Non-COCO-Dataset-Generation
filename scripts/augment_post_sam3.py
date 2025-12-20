"""
Use colour_augment module to augment images in a directory
and save augmented images and labels to an output directory.
"""

from pathlib import Path
import sys
import shutil
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.colour_augment import colour_augment
from src.dust_add import add_dust


def augment_directory(
    images_dir: Path,
    labels_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    copies_per_image: int = 1,
    add_dust_flag: bool = False,
    dust_intensity: float = 0.5,
):
    """
    Apply colour augmentation to all images in a directory.

    Labels are copied unchanged.
    """

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.jpg"))

    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    print(f"Augmenting {len(image_paths)} images")

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"Skipping {img_path.name} (no label)")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path.name}")
            continue

        # Copy original (optional but useful)
        shutil.copy(img_path, out_images_dir / img_path.name)
        shutil.copy(label_path, out_labels_dir / label_path.name)

        for i in range(copies_per_image):
            aug_img = colour_augment(img)

            if add_dust_flag:
                aug_img = add_dust(aug_img, intensity=dust_intensity)


            new_name = f"{img_path.stem}_aug{i}"
            out_img_path = out_images_dir / f"{new_name}.jpg"
            out_lbl_path = out_labels_dir / f"{new_name}.txt"

            cv2.imwrite(str(out_img_path), aug_img)
            shutil.copy(label_path, out_lbl_path)

    print("Colour augmentation complete")


def main():
    augment_directory(
        images_dir=Path("data/sam_output/images"),
        labels_dir=Path("data/sam_output/labels"),
        out_images_dir=Path("data/augmented/images"),
        out_labels_dir=Path("data/augmented/labels"),
        copies_per_image=2,
        add_dust_flag=True,
        dust_intensity=0.3,
    )


if __name__ == "__main__":
    main()
