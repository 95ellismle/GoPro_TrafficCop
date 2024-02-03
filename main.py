import cv2
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from storage import STORAGE_PATH


cap = cv2.VideoCapture(str(STORAGE_PATH / 'video/traffic_lights.360'))

i = 0
while True:
    ret, frame = cap.read()
    i += 1
    cv2.imshow(f"frame_{i}", frame)
    cv2.waitKey(0)


