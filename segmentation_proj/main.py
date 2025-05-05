import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (you can use v5 too with a custom loader)
model = YOLO("yolov8n.pt")  # Lightweight model

# Define HSV bounds for rot detection (tune if needed)
lower = np.array([10, 50, 20])
upper = np.array([30, 255, 255])
kernel = np.ones((5, 5), np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()

print("✅ Webcam running... Close the window or press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)[0]
    frame_copy = frame.copy()
    detected_count = 0

    for r in results.boxes:
        cls = int(r.cls[0])
        label = model.names[cls]

        # Filter for fruits or leaves only
        if label.lower() in ["apple", "banana", "orange", "leaf"]:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            # Detect rot in ROI
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    dx, dy, dw, dh = cv2.boundingRect(cnt)
                    cv2.rectangle(roi, (dx, dy), (dx + dw, dy + dh), (0, 0, 255), 2)
                    detected_count += 1

            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

    # Display total count
    cv2.putText(frame_copy, f"Rotten spots: {detected_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Rotten Detection (Close window to stop)", frame_copy)

    # Stop if 'q' is pressed or window is closed
    if cv2.getWindowProperty("Rotten Detection (Close window to stop)", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed and cleaned up.")