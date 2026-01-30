"""
Compare performance between SAM3 segmentation and YOLO inference
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_timing_file(filepath):
    """Read timing data from file and return list of times in milliseconds"""
    times = []
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                time_ms = float(parts[2])
                times.append(time_ms)
    return times


def calculate_stats(times):
    """Calculate statistics from timing data"""
    if not times:
        return None
    
    return {
        'count': len(times),
        'total': sum(times),
        'average': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'median': sorted(times)[len(times) // 2]
    }


def main():
    sam3_timing_file = PROJECT_ROOT / "data" / "sam3_timing.txt"
    yolo_timing_file = PROJECT_ROOT / "data" / "yolo_timing.txt"
    
    # Check if files exist
    if not sam3_timing_file.exists():
        print(f"SAM3 timing file not found: {sam3_timing_file}")
        return
    
    if not yolo_timing_file.exists():
        print(f"YOLO timing file not found: {yolo_timing_file}")
        return
    
    # Read timing data
    sam3_times = read_timing_file(sam3_timing_file)
    yolo_times = read_timing_file(yolo_timing_file)
    
    # Calculate statistics
    sam3_stats = calculate_stats(sam3_times)
    yolo_stats = calculate_stats(yolo_times)
    
    if not sam3_stats or not yolo_stats:
        print("Error: Unable to calculate statistics from timing files")
        return
    
    # Print comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: SAM3 vs YOLO")
    print("="*70)
    
    print("\n" + "-"*70)
    print("SAM3 SEGMENTATION STATISTICS")
    print("-"*70)
    print(f"  Total Frames:        {sam3_stats['count']}")
    print(f"  Total Time:          {sam3_stats['total']/1000:.2f} seconds")
    print(f"  Average Time:        {sam3_stats['average']:.2f} ms/frame")
    print(f"  Min Time:            {sam3_stats['min']:.2f} ms/frame")
    print(f"  Max Time:            {sam3_stats['max']:.2f} ms/frame")
    print(f"  Median Time:         {sam3_stats['median']:.2f} ms/frame")
    
    print("\n" + "-"*70)
    print("YOLO INFERENCE STATISTICS")
    print("-"*70)
    print(f"  Total Frames:        {yolo_stats['count']}")
    print(f"  Total Time:          {yolo_stats['total']/1000:.2f} seconds")
    print(f"  Average Time:        {yolo_stats['average']:.2f} ms/frame")
    print(f"  Min Time:            {yolo_stats['min']:.2f} ms/frame")
    print(f"  Max Time:            {yolo_stats['max']:.2f} ms/frame")
    print(f"  Median Time:         {yolo_stats['median']:.2f} ms/frame")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    speed_ratio = sam3_stats['average'] / yolo_stats['average']
    
    if speed_ratio > 1:
        print(f"  YOLO is {speed_ratio:.2f}x FASTER than SAM3")
    else:
        print(f"  SAM3 is {1/speed_ratio:.2f}x FASTER than YOLO")
    
    print(f"\n  Average Time Difference: {abs(sam3_stats['average'] - yolo_stats['average']):.2f} ms/frame")
    
    total_time_diff = abs(sam3_stats['total'] - yolo_stats['total']) / 1000
    print(f"  Total Time Difference:   {total_time_diff:.2f} seconds")
    
    # Throughput comparison
    sam3_fps = 1000 / sam3_stats['average']
    yolo_fps = 1000 / yolo_stats['average']
    
    print(f"\n  SAM3 Throughput:     {sam3_fps:.2f} FPS")
    print(f"  YOLO Throughput:     {yolo_fps:.2f} FPS")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if yolo_stats['average'] < sam3_stats['average']:
        print("  ✓ YOLO is more efficient for real-time detection")
        print("  ✓ Recommended for deployment scenarios requiring speed")
    else:
        print("  ✓ SAM3 is more efficient in this comparison")
        print("  ✓ Note: SAM3 provides segmentation, YOLO provides detection")
    
    print("\n  Note: SAM3 generates training data, YOLO uses it for inference.")
    print("  Different tasks with different computational requirements.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
