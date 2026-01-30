"""
YOLO testing and inference with improved visualization and performance metrics
"""

from pathlib import Path
import sys
import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Class colors (BGR format for OpenCV) - matching SAM3 colors
CLASS_COLORS = {
    0: (0, 255, 255),    # Yellow keychain - Cyan
    1: (255, 0, 255),    # Yellow rubberduck - Magenta
    2: (255, 255, 0),    # White swan figurine - Yellow
}

CLASS_NAMES = {
    0: "Yellow keychain",
    1: "Yellow rubberduck", 
    2: "White swan figurine",
}


def annotate_yolo_results(image, results, processing_time=0.0):
    """
    Annotate image with YOLO detection results using improved visualization
    """
    height, width = image.shape[:2]
    
    # Extract boxes, classes, and confidences
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            # Get color for this class
            color = CLASS_COLORS.get(cls, (0, 255, 0))
            
            # Draw thicker rectangle
            thickness = 3
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label with class name and confidence
            label_text = f"{CLASS_NAMES.get(cls, f'Class {cls}')}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )
            
            # Draw label background
            label_y1 = max(y1 - text_height - 10, 0)
            label_y2 = y1
            label_x1 = x1
            label_x2 = x1 + text_width + 10
            
            overlay = image.copy()
            cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
            
            # Draw label text
            cv2.putText(
                image, 
                label_text, 
                (x1 + 5, y1 - 5), 
                font, 
                font_scale, 
                (255, 255, 255), 
                font_thickness
            )
    
    # Add timing overlay in top right
    timing_text = f"YOLO: {processing_time*1000:.1f}ms/frame"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(
        timing_text, font, font_scale, font_thickness
    )
    
    # Draw semi-transparent background
    padding = 10
    bg_x1 = width - text_width - padding * 2
    bg_y1 = padding
    bg_x2 = width - padding
    bg_y2 = padding + text_height + padding
    
    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Draw timing text
    cv2.putText(
        image,
        timing_text,
        (bg_x1 + padding, bg_y1 + text_height + 5),
        font,
        font_scale,
        (255, 165, 0),  # Orange color for YOLO
        font_thickness
    )
    
    return image


def test_yolo_and_create_video(model_path, test_images_dir, output_video_path, timing_file, fps=30):
    """
    Run YOLO inference on test images and create annotated video using timing data
    """
    model_path = Path(model_path)
    test_images_dir = Path(test_images_dir)
    output_video_path = Path(output_video_path)
    timing_file = Path(timing_file)
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train the model first by running yolo_training.py")
        return
    
    # Load YOLO model
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))
    
    # Get test images - look recursively for all jpg files
    image_files = sorted(test_images_dir.rglob("*.jpg"))
    
    if not image_files:
        print(f"No test images found in {test_images_dir}")
        return
    
    print(f"Found {len(image_files)} test images")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    print(f"Running YOLO inference and creating video...")
    
    total_inference_time = 0
    total_frames = 0
    timing_records = []

    dummy = cv2.imread(str(image_files[0]))

    for _ in range(5):
        _ = model(dummy, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

        
    for idx, image_file in enumerate(image_files, 1):
        # Read image (NOT timed – correct)
        image = cv2.imread(str(image_file))
        
        # --- YOLO pure inference timing ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.time()
        results = model(image, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_time = time.time() - t0
        # --------------------------------

        total_inference_time += inference_time
        total_frames += 1
        
        # Get frame name (handle nested paths to match sam3_timing.txt)
        try:
            rel_path = image_file.relative_to(test_images_dir)
            frame_name = "_".join(rel_path.with_suffix("").parts)
        except:
            frame_name = image_file.stem
        
        timing_records.append((frame_name, inference_time))
        
        # Annotate image (NOT timed – correct)
        annotated_image = annotate_yolo_results(image, results, inference_time)
        
        # Write frame (NOT timed – correct)
        video_writer.write(annotated_image)
        
        if idx % 10 == 0 or idx == len(image_files):
            print(f"  Processed {idx}/{len(image_files)} frames")
    
    video_writer.release()
    
    avg_inference_time = total_inference_time / total_frames
    
    # Save timing data
    timing_file = PROJECT_ROOT / "data" / "yolo_timing.txt"
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(timing_file, "w") as f:
        f.write("Frame,Time(seconds),Time(ms)\n")
        for frame_name, frame_time in timing_records:
            f.write(f"{frame_name},{frame_time:.6f},{frame_time*1000:.2f}\n")
    
    print(f"\nYOLO testing complete!")
    print(f"Video saved to: {output_video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Average inference time: {avg_inference_time*1000:.2f}ms/frame")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Timing data saved to: {timing_file}")
    



def main():
    # Find the trained model (handle nested runs/detect/runs/detect structure)
    possible_dirs = [
        PROJECT_ROOT / "runs" / "detect" / "runs" / "detect",  # Nested structure
        PROJECT_ROOT / "runs" / "detect",  # Direct structure
    ]
    
    model_dirs = []
    for runs_dir in possible_dirs:
        if runs_dir.exists():
            found = sorted(runs_dir.glob("3class_yolov8n*"), key=lambda x: x.stat().st_mtime, reverse=True)
            model_dirs.extend(found)
    
    if not model_dirs:
        print("No trained model found. Please run yolo_training.py first.")
        print(f"Searched in: {[str(d) for d in possible_dirs]}")
        return
    
    # Use the most recent model
    model_path = model_dirs[0] / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"Model weights not found at: {model_path}")
        return
    
    # Test on pre_sam images (same images SAM3 used)
    test_images_dir = PROJECT_ROOT / "data" / "pre_sam" / "images"
    output_video = PROJECT_ROOT / "data" / "yolo_test_output.mp4"
    timing_file = PROJECT_ROOT / "data" / "yolo_timing.txt"
    
    # Get FPS (same as SAM3 video)
    video_dir = PROJECT_ROOT / "videos"
    video_files = list(video_dir.glob("*.mp4"))
    
    fps = 30  # Default FPS
    if video_files:
        cap = cv2.VideoCapture(str(video_files[0]))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fps = original_fps / 5  # Adjust for frame stride
        print(f"Using FPS: {fps}")
    
    test_yolo_and_create_video(model_path, test_images_dir, output_video, timing_file, fps=fps)


if __name__ == "__main__":
    main()