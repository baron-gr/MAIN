import os
import time
import uuid
import cv2

IMAGES_PATH = os.path.join('data', 'images')
number_images = 30

cap = cv2.VideoCapture(1)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Labelme

import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt






# import albumentations as alb
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import load_model

