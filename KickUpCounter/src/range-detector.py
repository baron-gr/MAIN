import cv2
import argparse
import numpy as np

def nothing(x):
    pass

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', help='Path to the video file')
args = parser.parse_args()

# Load video or webcam
video_source = args.video if args.video else 0
cap = cv2.VideoCapture(video_source)

# Create a window
cv2.namedWindow("Trackbars")

# Create trackbars for color range
cv2.createTrackbar("LH", "Trackbars", 0, 255, nothing)  # Lower Hue
cv2.createTrackbar("LS", "Trackbars", 0, 255, nothing)  # Lower Saturation
cv2.createTrackbar("LV", "Trackbars", 0, 255, nothing)  # Lower Value
cv2.createTrackbar("UH", "Trackbars", 255, 255, nothing)  # Upper Hue
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)  # Upper Saturation
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)  # Upper Value

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (600, 400))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    lh = cv2.getTrackbarPos("LH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uh = cv2.getTrackbarPos("UH", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")

    lower_bound = np.array([lh, ls, lv])
    upper_bound = np.array([uh, us, uv])

    # Create a mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply the mask
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Lower HSV: {lower_bound}")
print(f"Upper HSV: {upper_bound}")

cap.release()
cv2.destroyAllWindows()