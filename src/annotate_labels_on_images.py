"""
this takes images, and their corresponding labels, and annotates the labels on the images
this is will help to visually verify the labels
because the labels are geenrated artifically, there may be errors
it is recommended to randomly check some of the output images to verify the labels
"""

import cv2
from pathlib import Path


def annotate_labels_on_images(labels_dir: str = "data/sam_output/labels", images_dir: str = "data/sam_output/images", output_dir: str = "data/sam_output_annotated"):
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = 0

    for label_file in labels_dir.glob("*.txt"):
        image_file = images_dir / f"{label_file.stem}.jpg"  # assuming .jpg images
        if not image_file.exists():
            print(f"Image file not found for label: {label_file.name}")
            continue
        # Read image
        image = cv2.imread(str(image_file))
        height, width, _ = image.shape
        # Read labels
        with open(label_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Invalid label format in {label_file.name}: {line.strip()}")
                continue
            class_id, x_center, y_center, w, h = map(float, parts)
            # Convert to pixel coordinates
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put class id text
            cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # Save annotated image
        output_path = output_dir / f"{label_file.stem}_annotated.jpg"
        cv2.imwrite(str(output_path), image)
        annotated += 1
    

    print(f"Annotated {annotated} images with labels.")

if __name__ == "__main__":
    annotate_labels_on_images()

    
