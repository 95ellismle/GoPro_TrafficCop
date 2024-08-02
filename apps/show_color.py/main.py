import numpy as np
import cv2 as cv


MIN_HUE = 85
MAX_HUE = 105

img = np.ones((500, 500, 3), dtype=np.uint8)
img[:, :, 1] = 255
img[:, :, 2] = 255

for i, col in zip(np.arange(500), np.linspace(MIN_HUE, MAX_HUE, 500)):
    print(i, col)
    img[:, i, 0] = int(col)

img = cv.cvtColor(img, cv.COLOR_HSV2RGB)

cv.imshow("BOB", img)
cv.waitKey(0)
cv.destroyAllWindows()

