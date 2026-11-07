import cv2
import argparse
from ultralytics import YOLO


def detect_humans_in_video(video_path, model_name='yolov8n.pt', confidence_threshold=0.5):
    """
    Detect humans in a video and display results in real-time.

    Args:
        video_path: Path to the video file
        model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
        confidence_threshold: Minimum confidence for detections (0-1)
    """
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
    print("Press 'q' to quit, 'p' to pause/resume")

    frame_count = 0
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()

            if not ret:
                print("End of video or error reading frame")
                break

            frame_count += 1

            # Detect only humans (class 0)
            results = model(
                frame, classes=[0], conf=confidence_threshold, verbose=False)

            # Annotate frame
            annotated_frame = results[0].plot()

            # Add frame counter and human count
            human_count = len(results[0].boxes)
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Humans: {human_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Human Detection', annotated_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('p'):
            paused = not paused
            status = "Paused" if paused else "Resumed"
            print(f"{status} at frame {frame_count}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(
        description='Detect humans in video using YOLO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py video.mp4
  python script.py video.mp4 --model yolov8s.pt
  python script.py video.mp4 --confidence 0.7
        """
    )

    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold 0-1 (default: 0.5)')

    args = parser.parse_args()

    detect_humans_in_video(args.video_path, args.model, args.confidence)


if __name__ == '__main__':
    main()
