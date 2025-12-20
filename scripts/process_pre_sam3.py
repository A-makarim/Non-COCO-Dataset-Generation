"""
Sam3 video propogation is computationally expensive. This script extracts frames
Thus a good comprmise 
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video_to_images import extract_frames

VIDEO_DIR = "videos"
OUT_DIR = "data/pre_sam/images"

FRAME_STRIDE = 5 # Process every 5th frame
RESIZE_WIDTH = 640 # None to disable

def main():
    video_dir = Path(VIDEO_DIR)
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Processing videos BEFORE SAM3")
    print(f"Videos: {VIDEO_DIR}")
    print(f"Output: {OUT_DIR}\n")

    patterns = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV")
    files = []
    for pat in patterns:
        files.extend(sorted(video_dir.glob(pat)))

    if not files:
        print(f"No video files found in {video_dir}. Nothing to process.")
    else:
        for video in files:
            video_out = out_root / video.stem
            video_out.mkdir(parents=True, exist_ok=True)
            print(f"Processing {video.name} -> {video_out}")
            extract_frames(
                video_path=video,
                out_dir=video_out,
                frame_stride=FRAME_STRIDE,
                resize_width=RESIZE_WIDTH,
            )

    print("\nPre-SAM processing complete")

if __name__ == "__main__":
    main()
