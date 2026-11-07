import cv2
import argparse
import os
from datetime import datetime
from ultralytics import YOLO


def detect_human_appearances(video_path, output_dir='output', model_name='yolov8n.pt',
                             confidence_threshold=0.5, sample_rate=5):
    """
    Detect when new humans appear in video and save screenshots.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save screenshots
        model_name: YOLO model to use
        confidence_threshold: Minimum confidence for detections (0-1)
        sample_rate: Process every Nth frame (default: 5)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Sample rate: Processing 1 out of every {sample_rate} frames")
    print(f"Output directory: {output_dir}/")
    print("-" * 60)

    frame_count = 0
    processed_count = 0
    previous_human_count = 0
    screenshot_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Skip frames based on sample rate
        if frame_count % sample_rate != 0:
            continue

        processed_count += 1

        # Detect only humans (class 0)
        results = model(
            frame, classes=[0], conf=confidence_threshold, verbose=False)
        current_human_count = len(results[0].boxes)

        # Check if human count increased
        if current_human_count > previous_human_count:
            # Calculate timestamp in video
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_sec = timestamp_ms / 1000
            hours = int(timestamp_sec // 3600)
            minutes = int((timestamp_sec % 3600) // 60)
            seconds = int(timestamp_sec % 60)
            milliseconds = int((timestamp_sec % 1) * 1000)

            # Create filename with structured timestamp format
            # Format: humans_COUNT_at_HH_MM_SS_MS_FRAME.jpg
            filename = f"humans_{current_human_count}_at_{hours:02d}_{minutes:02d}_{seconds:02d}_{milliseconds:03d}_{frame_count:06d}.jpg"
            filepath = os.path.join(output_dir, filename)

            # Annotate and save frame
            annotated_frame = results[0].plot()

            # Add info overlay
            cv2.putText(annotated_frame, f"Timestamp: {hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Humans: {current_human_count}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imwrite(filepath, annotated_frame)
            screenshot_count += 1

            # Print update
            print(f"[Frame {processed_count}/{total_frames//sample_rate}] "
                  f"Human count: {previous_human_count} → {current_human_count} | "
                  f"Screenshot saved: {filename}")

            previous_human_count = current_human_count

        # Update previous count even if it decreased (person left)
        if current_human_count < previous_human_count:
            print(f"[Frame {processed_count}/{total_frames//sample_rate}] "
                  f"Human count decreased: {previous_human_count} → {current_human_count}")
            previous_human_count = current_human_count

    # Cleanup
    cap.release()

    print("-" * 60)
    print(f"Processing complete!")
    print(f"Total frames in video: {total_frames}")
    print(f"Frames processed: {processed_count} (every {sample_rate} frames)")
    print(f"Screenshots saved: {screenshot_count}")
    print(f"Output location: {os.path.abspath(output_dir)}/")


def main():
    parser = argparse.ArgumentParser(
        description='Detect human appearances in video and save screenshots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py video.mp4
  python script.py video.mp4 --output screenshots --sample-rate 10
  python script.py video.mp4 --model yolov8s.pt --confidence 0.7
  python script.py video.mp4 --sample-rate 30  # Process every 30th frame
        """
    )

    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for screenshots (default: output)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold 0-1 (default: 0.5)')
    parser.add_argument('--sample-rate', type=int, default=5,
                        help='Process every Nth frame (default: 5)')

    args = parser.parse_args()

    detect_human_appearances(
        args.video_path,
        args.output,
        args.model,
        args.confidence,
        args.sample_rate
    )


if __name__ == '__main__':
    main()
