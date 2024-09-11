import numpy as np
import cv2 as cv


MIN_HUE = 0
MAX_HUE = 255

img = np.ones((255, 255, 3), dtype=np.uint8)
img[:, :, 1] = 255
img[:, :, 2] = 255

for i, col in zip(np.arange(255), np.linspace(MIN_HUE, MAX_HUE, 255)):
    print(i, col)
    img[:, i, 0] = int(col)

img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
import matplotlib.pyplot as plt
plt.imshow(img); plt.colorbar(); plt.show()

import ipdb; ipdb.set_trace()

cv.imshow("BOB", img)
cv.waitKey(0)
cv.destroyAllWindows()

