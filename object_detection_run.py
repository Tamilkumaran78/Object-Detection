from ultralytics import YOLO
import cv2

def main():
    # Load YOLO model
    # Replace 'yolov8n.pt' with your trained model file if you have one, e.g., 'best.pt'
    model = YOLO('model.pt')

    # Ask user what mode to run
    print("Select mode:")
    print("1 - Image")
    print("2 - Video")
    print("3 - Webcam")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == '1':
        image_path = input("Enter the image file path: ").strip()
        detect_in_image(model, image_path)

    elif choice == '2':
        video_path = input("Enter the video file path: ").strip()
        detect_in_video(model, video_path)

    elif choice == '3':
        detect_from_webcam(model)

    else:
        print("Invalid choice! Please enter 1, 2, or 3.")


def detect_in_image(model, image_path):
    """
    Detect objects in a single image.
    """
    print(f"Running detection on image: {image_path}")
    results = model(image_path, show=True)  # show=True displays image in window
    # Save result
    results[0].save(filename='result.jpg')
    print("Detection complete. Saved to result.jpg.")


def detect_in_video(model, video_path):
    """
    Detect objects in a video file.
    """
    print(f"Running detection on video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)

        # Draw results on frame
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow('YOLO Detection - Video', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video detection finished.")


def detect_from_webcam(model):
    """
    Detect objects using webcam.
    """
    print("Starting webcam detection. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)  # Change to other index if multiple webcams

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)

        # Draw results on frame
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow('YOLO Detection - Webcam', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection finished.")


if __name__ == '__main__':
    main()
