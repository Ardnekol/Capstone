import cv2
import math
import cvzone
import os
from ultralytics import YOLO

# Load YOLOv8 model with custom weights
yolo_model = YOLO("Weights/best.pt")  # Make sure this file exists

# Define class labels (update as per your dataset)
class_labels = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

# Path to media directory
media_dir = "Media"

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png")

# Get all image file names in the folder
image_files = [f for f in os.listdir(media_dir) if f.lower().endswith(image_extensions)]

if not image_files:
    print("[ERROR] No image files found in the Media directory.")
    exit()

# Loop through all images
for image_file in image_files:
    image_path = os.path.join(media_dir, image_file)
    img = cv2.imread(image_path)

    if img is None:
        print(f"[WARNING] Could not load image: {image_path}")
        continue

    # Perform object detection
    results = yolo_model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.3:
                label = class_labels[cls] if cls < len(class_labels) else f"Class {cls}"
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(img, f'{label} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

    # Display image
    cv2.imshow("Garbage Detection", img)
    print(f"[INFO] Showing: {image_file}")

    # Wait for key: Press 'q' to quit or any other key to show next
    key = cv2.waitKey(0)  # 0 = wait indefinitely
    if key == ord('q'):
        print("[INFO] Exiting early.")
        break

# Cleanup
cv2.destroyAllWindows()
