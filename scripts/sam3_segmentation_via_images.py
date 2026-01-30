import cv2
import torch
import time
from pathlib import Path
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


INPUT_IMG_DIR = "data/pre_sam/images"
OUT_IMG_DIR   = "data/sam_output/images"
OUT_LBL_DIR   = "data/sam_output/labels"

# Text prompts for different objects
TEXT_PROMPTS = [
    "Yellow keychain",        # class 0
    "yellow rubberduck",      # class 1
    "white swan figurine"     # class 2
]
CONF_THRESH = 0.3


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    Path(OUT_IMG_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUT_LBL_DIR).mkdir(parents=True, exist_ok=True)

    # MODEL LOAD IS NOW SAFE
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model)

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
    image_paths = []
    for pat in patterns:
        image_paths.extend(Path(INPUT_IMG_DIR).rglob(pat))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        raise RuntimeError(f"No images found in {INPUT_IMG_DIR}")

    print(f"Found {len(image_paths)} images")

    # Track timing for each frame
    timing_records = []

    for idx, img_path in enumerate(image_paths):
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        yolo_lines = []

        # Process each class prompt
        for class_id, prompt in enumerate(TEXT_PROMPTS):
            with torch.no_grad():
                frame_start_time = time.time()

                state = processor.set_image(pil_img)
                output = processor.set_text_prompt(
                    state=state,
                    prompt=prompt
                )

                frame_time = time.time() - frame_start_time




            boxes = output.get("boxes", [])
            scores = output.get("scores", [])

            for box, score in zip(boxes, scores):
                if score < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = box.tolist()
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        try:
            rel = img_path.relative_to(Path(INPUT_IMG_DIR))
            name = "_".join(rel.with_suffix("").parts)
        except Exception:
            name = img_path.stem

        cv2.imwrite(f"{OUT_IMG_DIR}/{name}.jpg", img)

        with open(f"{OUT_LBL_DIR}/{name}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

        # Record timing for this frame
        timing_records.append((name, frame_time))

        if idx % 25 == 0:
            print(f"[{idx}/{len(image_paths)}] processed")

    # Save timing data
    timing_file = Path("data/sam3_timing.txt")
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(timing_file, "w") as f:
        f.write("Frame,Time(seconds),Time(ms)\n")
        for frame_name, frame_time in timing_records:
            f.write(f"{frame_name},{frame_time:.6f},{frame_time*1000:.2f}\n")
    
    avg_time = sum(t for _, t in timing_records) / len(timing_records) if timing_records else 0
    print(f"\nSAM3 segmentation complete")
    print(f"Average time per frame: {avg_time*1000:.2f}ms")
    print(f"Timing data saved to: {timing_file}")


if __name__ == "__main__":
    main()
