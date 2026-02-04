"""
YOLOv8 Detection with Custom Annotations
- Calculates bounding box area
- Estimates rock weight based on scaling factor
- Thin, smaller annotations showing class, confidence, and weight
"""

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


# ============ CONFIGURATION ============
MODEL_PATH = "runs/detect/runs/detect/probe_rocks_yolov8n/weights/best.pt"
SOURCE = "videos_rock/Video Project.mp4"  # video/image/folder path
CONF_THRESHOLD = 0.2
IMGSZ = 640
DEVICE = 0  # 0 for GPU, 'cpu' for CPU

# Weight estimation parameters (adjust manually based on camera distance)
# Formula: weight_kg = area_pixels * SCALE_FACTOR
# This is a rough approximation - tune based on known object distances
SCALE_FACTOR = 0.0001  # kg per pixel² . can later be calibrated using known objects in the scene

# Annotation styling
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1  # smaller text
FONT_THICKNESS = 3  # thinner
BOX_THICKNESS = 3  # thinner boxes
TEXT_BG_ALPHA = 0.8  # semi-transparent background

# Class colors (BGR format)
COLORS = {
    0: (0, 255, 0),    # probe - green
    1: (0, 165, 255),  # mars-rock - orange
}
# ========================================


def estimate_weight(box_area_px, scale_factor=SCALE_FACTOR):
    """
    Estimate weight in kg based on bounding box area.
    
    Args:
        box_area_px: Area of bounding box in pixels²
        scale_factor: Manually tuned factor (kg/pixel²)
    
    Returns:
        Estimated weight in kg
    """
    return box_area_px * scale_factor


def draw_custom_annotations(frame, results, scale_factor=SCALE_FACTOR):
    """
    Draw custom thin annotations with weight estimates.
    
    Args:
        frame: Video frame (numpy array)
        results: YOLO prediction results
        scale_factor: Weight scaling factor
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates and info
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]
            
            # Calculate area and estimate weight
            box_width = x2 - x1
            box_height = y2 - y1
            area_px = box_width * box_height
            weight_kg = estimate_weight(area_px, scale_factor)
            
            # Get color for this class
            color = COLORS.get(cls, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            
            # Prepare label text
            label = f"{class_name} {conf:.2f} | {weight_kg:.3f}kg"
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, FONT, FONT_SCALE, FONT_THICKNESS
            )
            
            # Draw semi-transparent background for text
            overlay = annotated.copy()
            cv2.rectangle(
                overlay,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w + 4, y1),
                color,
                -1
            )
            cv2.addWeighted(overlay, TEXT_BG_ALPHA, annotated, 1 - TEXT_BG_ALPHA, 0, annotated)
            
            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - baseline - 2),
                FONT,
                FONT_SCALE,
                (255, 255, 255),  # white text
                FONT_THICKNESS,
                cv2.LINE_AA
            )
    
    return annotated


def process_video(model, source, output_dir="runs/detect/custom_annotated"):
    """Process video with custom annotations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video writer
    source_name = Path(source).stem
    output_video = output_path / f"{source_name}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames...")
    print(f"Scale factor: {SCALE_FACTOR} kg/pixel²")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO inference
        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False
        )
        
        # Draw custom annotations
        annotated_frame = draw_custom_annotations(frame, results, SCALE_FACTOR)
        
        # Write frame
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    print(f"\nDone! Output saved to: {output_video}")


def process_image(model, source, output_dir="runs/detect/custom_annotated"):
    """Process single image with custom annotations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    frame = cv2.imread(source)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {source}")
    
    print(f"Processing image: {source}")
    print(f"Scale factor: {SCALE_FACTOR} kg/pixel²")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    
    # Run YOLO inference
    results = model.predict(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False
    )
    
    # Draw custom annotations
    annotated_frame = draw_custom_annotations(frame, results, SCALE_FACTOR)
    
    # Save
    source_name = Path(source).stem
    output_image = output_path / f"{source_name}_annotated.jpg"
    cv2.imwrite(str(output_image), annotated_frame)
    
    print(f"Done! Output saved to: {output_image}")


def main():
    # Auto-detect device
    device = DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Check if source is video or image
    source_path = Path(SOURCE)
    if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        process_video(model, SOURCE)
    elif source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(model, SOURCE)
    else:
        raise ValueError(f"Unsupported file format: {source_path.suffix}")


if __name__ == "__main__":
    main()
