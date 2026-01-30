import subprocess
import sys
from pathlib import Path

def run_script(path):
    print(f"\n--- Running {path} ---")
    subprocess.run(
        [sys.executable, path],
        check=True
    )

def main():
    run_script("scripts/process_pre_sam3.py")
    run_script("scripts/sam3_segmentation_via_images.py")
    run_script("scripts/refine_pre_sam3.py")
    run_script("scripts/augment_post_sam3.py")
    run_script("scripts/visualize_post_sam3.py")
    run_script("scripts/split_sam3_data.py")
    run_script("scripts/yolo_training.py")
    run_script("scripts/yolo_testing.py")
    run_script("scripts/create_sam3_video.py")
    run_script("scripts/compare_performance.py")

if __name__ == "__main__":
    main()
