import cv2 as cv
import numpy as np
from pathlib import Path

from storage import STORAGE_PATH


cap = cv.VideoCapture(str(STORAGE_PATH / 'video/traffic_lights.360'))

ret, frame = cap.read()


i = 1
cv.imshow(f"frame_{i}", frame)

k = cv.waitKey(0)
