import cv2
import argparse
from collections import deque
import imutils
from imutils.video import VideoStream
import numpy as np
import time

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', help='path to the video file')
parser.add_argument('-b', '--buffer', type=int, default=64, help='max buffer size')
args = vars(parser.parse_args())

# ball colour boundaries
whiteLower = 
whiteUpper = 
pts = deque(maxlen=args['buffer'])

if not args.get('video', False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args['video'])

time.sleep(2.0)

while True:
    frame = vs.read()
    
    frame = frame[1] if args.get('video', False) else frame
    
    if frame is None:
        break
    
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, whiteLower, whiteUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    