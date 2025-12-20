"""
used to generate images for sam3 segmentation from image file which is much faster than doing it on video files
"""


import cv2
from pathlib import Path

def extract_frames(video_path, out_dir, frame_stride=5, resize_width=None):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        idx += 1
        if idx % frame_stride != 0:
            continue

        if resize_width:
            h, w = frame.shape[:2]
            scale = resize_width / w
            frame = cv2.resize(
                frame,
                (resize_width, int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )

        cv2.imwrite(str(out_dir / f"frame_{saved:06d}.jpg"), frame)
        saved += 1

    cap.release()
    print(f"Saved {saved} frames")
