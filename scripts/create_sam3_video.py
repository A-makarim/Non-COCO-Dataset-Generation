"""
Create annotated video from SAM3 output with improved visualization and performance metrics
"""

from pathlib import Path
import sys
import cv2
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Class colors (BGR format for OpenCV)
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


def annotate_image_with_labels(image, label_file, show_timing=True, processing_time=0.0):
    """
    Annotate image with bounding boxes and class labels with improved visualization
    """
    height, width = image.shape[:2]
    
    if label_file.exists():
        with open(label_file, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id, x_center, y_center, w, h = map(float, parts)
            class_id = int(class_id)
            
            # Convert to pixel coordinates
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            
            # Get color for this class
            color = CLASS_COLORS.get(class_id, (0, 255, 0))
            
            # Draw thicker rectangle with rounded corners effect
            thickness = 3
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw semi-transparent filled rectangle for label background
            label_text = CLASS_NAMES.get(class_id, f"Class {class_id}")
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
    if show_timing:
        timing_text = f"SAM3: {processing_time*1000:.1f}ms/frame"
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
            (0, 255, 0),
            font_thickness
        )
    
    return image


def create_video_from_frames(images_dir, labels_dir, output_video_path, timing_file, fps=30):
    """
    Create video from annotated frames using timing data from file
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_video_path = Path(output_video_path)
    timing_file = Path(timing_file)
    
    # Read timing data from file
    timing_dict = {}
    if timing_file.exists():
        with open(timing_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    frame_name = parts[0]
                    time_seconds = float(parts[1])
                    timing_dict[frame_name] = time_seconds
    else:
        print(f"Warning: Timing file not found: {timing_file}")
    
    # Get all image files sorted by name
    image_files = sorted(images_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    print(f"Creating video with {len(image_files)} frames at {fps} FPS...")
    
    total_processing_time = 0
    for idx, image_file in enumerate(image_files, 1):
        # Measure annotation time
        start_time = time.time()
        
        # Read image
        image = cv2.imread(str(image_file))
        
        # Find corresponding label file
        label_file = labels_dir / f"{image_file.stem}.txt"
        
        # Get actual processing time from timing file
        processing_time = timing_dict.get(image_file.stem, 0.033)
        
        # Annotate image
        annotated_image = annotate_image_with_labels(
            image, 
            label_file, 
            show_timing=True,
            processing_time=processing_time
        )
        
        annotation_time = time.time() - start_time
        total_processing_time += annotation_time
        
        # Write frame to video
        video_writer.write(annotated_image)
        
        if idx % 10 == 0 or idx == len(image_files):
            print(f"  Processed {idx}/{len(image_files)} frames")
    
    video_writer.release()
    
    avg_time = total_processing_time / len(image_files)
    print(f"\nVideo saved to: {output_video_path}")
    print(f"Total frames: {len(image_files)}")
    print(f"FPS: {fps}")
    print(f"Average annotation time: {avg_time*1000:.2f}ms/frame")


def main():
    images_dir = PROJECT_ROOT / "data" / "sam_output" / "images"
    labels_dir = PROJECT_ROOT / "data" / "sam_output" / "labels"
    output_video = PROJECT_ROOT / "data" / "sam3_annotated_output.mp4"
    timing_file = PROJECT_ROOT / "data" / "sam3_timing.txt"
    
    # Get FPS from original video if available
    video_dir = PROJECT_ROOT / "videos"
    video_files = list(video_dir.glob("*.mp4"))
    
    fps = 30  # Default FPS
    if video_files:
        cap = cv2.VideoCapture(str(video_files[0]))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        # Adjust for frame stride used in extraction
        fps = original_fps / 5  # FRAME_STRIDE = 5 in process_pre_sam3.py
        print(f"Original video FPS: {original_fps}, Adjusted FPS: {fps}")
    
    create_video_from_frames(images_dir, labels_dir, output_video, timing_file, fps=fps)
    print("\nSAM3 annotated video created successfully!")


if __name__ == "__main__":
    main()
