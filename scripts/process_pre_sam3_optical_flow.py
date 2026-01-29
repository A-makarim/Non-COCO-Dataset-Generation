"""
Optical flow-based adaptive frame extraction for SAM3 video processing.
Uses Farneback optical flow to detect actual motion/geometry changes,
ignoring lighting variations and compression artifacts.

This is the industry-standard approach for adaptive video sampling.
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_DIR = "videos"
OUT_DIR = "data/pre_sam_optical/images"

RESIZE_WIDTH = 640  # None to disable
MOTION_THRESHOLD = 0.6  # Motion score threshold (tune based on your video)
MIN_FRAME_INTERVAL = 2  # Minimum frames between saved frames
MAX_FRAME_INTERVAL = 30  # Maximum frames between saved frames (ensure coverage)


def calculate_optical_flow_motion(prev_gray, curr_gray):
    """
    Calculate motion score using Farneback optical flow.
    Returns a scalar representing the average motion magnitude.
    
    Higher values = more motion
    Lower values = static scene
    """
    # Farneback optical flow parameters
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, 
        curr_gray,
        None,
        pyr_scale=0.5,      # Image pyramid scale
        levels=3,           # Number of pyramid layers
        winsize=15,         # Averaging window size
        iterations=3,       # Iterations at each pyramid level
        poly_n=5,           # Size of pixel neighborhood
        poly_sigma=1.2,     # Gaussian standard deviation
        flags=0
    )
    
    # Convert flow to polar coordinates (magnitude and angle)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Motion score is simply the average magnitude
    motion_score = mag.mean()
    
    return motion_score, mag


def extract_frames_optical_flow(video_path, out_dir, motion_threshold=1.0,
                                 resize_width=None, min_interval=2, max_interval=30):
    """
    Extract frames adaptively based on optical flow motion detection.
    
    Args:
        video_path: Path to input video
        out_dir: Output directory for frames
        motion_threshold: Threshold for motion score
        resize_width: Width to resize frames to (None to disable)
        min_interval: Minimum frames between saved frames
        max_interval: Maximum frames between saved frames
    
    Returns:
        dict: Statistics about the extraction process
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    idx = 0
    saved = 0
    skipped = 0
    prev_gray = None
    last_saved_idx = -min_interval
    
    motion_scores = []
    start_time = time.time()
    
    print(f"Video properties: {total_frames} frames @ {fps:.2f} fps")
    print(f"Motion threshold: {motion_threshold}")
    print(f"Processing frames...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame if requested
        processed_frame = frame.copy()
        if resize_width:
            h, w = processed_frame.shape[:2]
            scale = resize_width / w
            processed_frame = cv2.resize(
                processed_frame,
                (resize_width, int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        
        # Convert to grayscale for optical flow
        curr_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        
        should_save = False
        motion_score = None
        
        # First frame is always saved
        if prev_gray is None:
            should_save = True
        else:
            # Calculate optical flow motion
            motion_score, _ = calculate_optical_flow_motion(prev_gray, curr_gray)
            motion_scores.append(motion_score)
            
            # Check frame interval constraints
            frames_since_last_save = idx - last_saved_idx
            
            # Decision logic (clean and simple)
            if frames_since_last_save < min_interval:
                # Too soon - skip
                should_save = False
            elif motion_score > motion_threshold:
                # Significant motion detected - save
                should_save = True
            elif frames_since_last_save >= max_interval:
                # Force save to ensure coverage
                should_save = True
            else:
                # Not enough motion and within interval - skip
                should_save = False
        
        if should_save:
            cv2.imwrite(str(out_dir / f"frame_{saved:06d}.jpg"), processed_frame)
            last_saved_idx = idx
            saved += 1
            
            if saved % 10 == 0:
                print(f"Progress: {idx}/{total_frames} frames processed, {saved} saved", end='\r')
        else:
            skipped += 1
        
        # Update previous frame
        prev_gray = curr_gray.copy()
        idx += 1
    
    cap.release()
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    stats = {
        'total_frames': total_frames,
        'frames_saved': saved,
        'frames_skipped': skipped,
        'compression_ratio': (1 - saved / total_frames) * 100 if total_frames > 0 else 0,
        'fps': fps,
        'processing_time': elapsed_time,
        'avg_motion': np.mean(motion_scores) if motion_scores else 0,
        'min_motion': np.min(motion_scores) if motion_scores else 0,
        'max_motion': np.max(motion_scores) if motion_scores else 0,
        'median_motion': np.median(motion_scores) if motion_scores else 0,
    }
    
    return stats


def print_stats(stats, video_name):
    """Print extraction statistics in a readable format."""
    print(f"\n{'='*60}")
    print(f"Statistics for: {video_name}")
    print(f"{'='*60}")
    print(f"Total frames in video:     {stats['total_frames']}")
    print(f"Frames saved:              {stats['frames_saved']}")
    print(f"Frames skipped:            {stats['frames_skipped']}")
    print(f"Compression ratio:         {stats['compression_ratio']:.2f}%")
    print(f"Original FPS:              {stats['fps']:.2f}")
    print(f"Effective FPS (saved):     {stats['frames_saved']/stats['total_frames']*stats['fps']:.2f}")
    print(f"Processing time:           {stats['processing_time']:.2f}s")
    if stats['avg_motion'] > 0:
        print(f"\nOptical Flow Motion metrics:")
        print(f"  Average motion:          {stats['avg_motion']:.4f}")
        print(f"  Median motion:           {stats['median_motion']:.4f}")
        print(f"  Min motion:              {stats['min_motion']:.4f}")
        print(f"  Max motion:              {stats['max_motion']:.4f}")
    print(f"{'='*60}\n")


def main():
    video_dir = Path(VIDEO_DIR)
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("OPTICAL FLOW ADAPTIVE FRAME EXTRACTION FOR SAM3")
    print("="*60)
    print(f"Videos directory:     {VIDEO_DIR}")
    print(f"Output directory:     {OUT_DIR}")
    print(f"Resize width:         {RESIZE_WIDTH}")
    print(f"Motion threshold:     {MOTION_THRESHOLD}")
    print(f"Min frame interval:   {MIN_FRAME_INTERVAL}")
    print(f"Max frame interval:   {MAX_FRAME_INTERVAL}")
    print("="*60 + "\n")

    patterns = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV")
    files = []
    for pat in patterns:
        files.extend(sorted(video_dir.glob(pat)))

    if not files:
        print(f"No video files found in {video_dir}. Nothing to process.")
        return

    all_stats = {}
    for video in files:
        video_out = out_root / video.stem
        video_out.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {video.name}")
        print(f"Output to: {video_out}\n")
        
        stats = extract_frames_optical_flow(
            video_path=video,
            out_dir=video_out,
            motion_threshold=MOTION_THRESHOLD,
            resize_width=RESIZE_WIDTH,
            min_interval=MIN_FRAME_INTERVAL,
            max_interval=MAX_FRAME_INTERVAL,
        )
        
        all_stats[video.name] = stats
        print_stats(stats, video.name)

    print("\n" + "="*60)
    print("SUMMARY OF ALL VIDEOS")
    print("="*60)
    total_original = sum(s['total_frames'] for s in all_stats.values())
    total_saved = sum(s['frames_saved'] for s in all_stats.values())
    print(f"Total frames across all videos: {total_original}")
    print(f"Total frames saved:             {total_saved}")
    print(f"Overall compression ratio:      {(1 - total_saved/total_original)*100:.2f}%")
    print("="*60)
    
    print("\nOptical flow-based pre-SAM processing complete!")


if __name__ == "__main__":
    main()
