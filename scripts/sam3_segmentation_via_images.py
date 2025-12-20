import os
import cv2
import torch
from pathlib import Path
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


INPUT_IMG_DIR = "data/pre_sam/images"
OUT_IMG_DIR   = "data/sam_output/images"
OUT_LBL_DIR   = "data/sam_output/labels"

TEXT_PROMPT = "yellow probe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.3


Path(OUT_IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(OUT_LBL_DIR).mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")

# Load SAM3 model ONCE
model = build_sam3_image_model(device=DEVICE)
processor = Sam3Processor(model)

patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
image_paths = []
for pat in patterns:
    image_paths.extend(Path(INPUT_IMG_DIR).rglob(pat))
image_paths = sorted(set(image_paths))

if not image_paths:
    raise RuntimeError(f"No images found in {INPUT_IMG_DIR}")

print(f"Found {len(image_paths)} images (recursive search)")

for idx, img_path in enumerate(image_paths):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    with torch.no_grad():
        state = processor.set_image(pil_img)
        output = processor.set_text_prompt(
            state=state,
            prompt=TEXT_PROMPT
        )

    boxes = output.get("boxes", [])
    scores = output.get("scores", [])

    yolo_lines = []

    for box, score in zip(boxes, scores):
        if score < CONF_THRESH:
            continue

        x1, y1, x2, y2 = box.tolist()

        # Clamp
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        # YOLO format (normalized)
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # create a unique name from the relative path to avoid collisions
    try:
        rel = img_path.relative_to(Path(INPUT_IMG_DIR))
        name = "_".join(rel.with_suffix("").parts)
    except Exception:
        name = img_path.stem

    # Save image (unchanged)
    cv2.imwrite(f"{OUT_IMG_DIR}/{name}.jpg", img)

    # Save label (empty file allowed)
    with open(f"{OUT_LBL_DIR}/{name}.txt", "w") as f:
        f.write("\n".join(yolo_lines))

    if idx % 25 == 0:
        print(f"[{idx}/{len(image_paths)}] processed")

print("✓ SAM3 segmentation complete")
print(f"Images → {OUT_IMG_DIR}")
print(f"Labels → {OUT_LBL_DIR}")
