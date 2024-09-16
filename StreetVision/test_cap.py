import torch
import cv2

# Set the path where the models should be cached
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', skip_validation=True)

# Access webcam
cap = cv2.VideoCapture(1)

# Optional: Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform detection
    results = model(frame)

    # Render results
    results.render()

    # Display the result
    cv2.imshow('YOLOv5 Webcam Detection', results.imgs[0])

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()