import cv2
import math
import cvzone
from ultralytics import YOLO

# Load YOLOv8 model with custom weights
model = YOLO("Weights/best.pt")  # Ensure the path is correct

# Class labels (should match your training labels)
classNames = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

# Initialize video capture (from file or use 0 for webcam)
video_path = "Media/garbage.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video is loaded correctly
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {video_path}")
    exit()

while True:
    success, img = cap.read()

    # If video ends or frame not read correctly, break loop
    if not success or img is None:
        print("[INFO] Video stream ended or failed to read frame.")
        break

    # Run YOLO object detection with streaming mode (faster for video)
    results = model(img, stream=True)

    # Iterate through each detection result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence and class ID
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Avoid index error if cls exceeds classNames
            label = classNames[cls] if cls < len(classNames) else f"Class {cls}"

            if conf > 0.1:  # Display only if confidence > 10%
                # Draw bounding box and label
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Show the current frame
    cv2.imshow("Garbage Detection", img)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting video display.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
