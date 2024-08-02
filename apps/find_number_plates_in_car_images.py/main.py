from pathlib import Path
import numpy as np
import cv2 as cv
from pprint import pprint as print
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import DBSCAN, HDBSCAN


DIR = Path('storage/img/training/cars/True')


def get_color_segmentations(img: np.ndarray, N: int):
    """Returns details of the color segmentation so the image can be reconstructed"""
    flattened = img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(flattened, N, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res


def segment_colors(img: np.ndarray, N: int):
    """Returns an image with colors reduced to the amount specified by N"""
    img = cv.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
    flattened_img = get_color_segmentations(img, N)
    ret = flattened_img.reshape((img.shape))
    return ret


def generate_yellow_contours(img: np.ndarray):
    mean_hue = np.mean(img[:, :, 0])
    mean_sat = np.mean(img[:, :, 1])
    mean_val = np.mean(img[:, :, 2])
    print(f"<H> = {mean_hue:.0f},  <S> = {mean_sat:.0f}, . <V> = {mean_val:.0f}")

    val_thresh = mean_val - 29
    val_thresh = 50 if val_thresh > 50 else val_thresh
    sat_thresh = mean_sat - 20
    sat_thresh = 50 if val_thresh > 50 else val_thresh

    yellows = img.copy()
    yellows[(yellows[:, :, 0] > 120) | (yellows[:, :, 0] < 40)  # Yellow hue
            | (yellows[:, :, 1] < sat_thresh)  # saturated colours
            | (yellows[:, :, 2] < val_thresh)  # brightish colours
    ] = [0, 0, 0]
    cv.imshow("Yellows", yellows)
    cv.waitKey(0)

    return

    yellows = cv.cvtColor(yellows, cv.COLOR_RGB2GRAY)
    yellows = cv.GaussianBlur(yellows,(5,5),0)
    _,yellows = cv.threshold(yellows,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)



def get_number_plate_box(image: np.ndarray):
    cv.imshow("original", image)
    image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    img =cv.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
    img_dim = np.mean([img.shape[1], img.shape[0]])
    img_area = img.shape[0] * img.shape[1]

    generate_yellow_contours(img)

#    kernel = np.ones((7,7),np.uint8)
#    yellows = cv.erode(yellows, kernel)
#    kernel = np.ones((13,13),np.uint8)
#    yellows = cv.dilate(yellows, kernel)
#    yellow_contours, _ = cv.findContours(yellows, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#    yellows = cv.cvtColor(yellows, cv.COLOR_GRAY2RGB)
#    yellows[:, :, 1] = 0
#
#    cnts = []
#    for contour in yellow_contours:
#        # Threshold size
#        rect = cv.minAreaRect(contour)
#        if cv.contourArea(contour) / img_area < 0.004:
#            continue
#        thresh = 0.02
#        if rect[1][0] / img_dim < thresh or rect[1][1] / img_dim < thresh:
#            continue
#        if rect[1][0] < 10 or rect[1][1] < 10:
#            continue
#
#        cnt_mask = np.zeros(img.shape[:2], dtype=np.uint8)
#        cv.drawContours(cnt_mask, [contour], -1, 1, -1)
#        cnt_mask = cnt_mask.astype(np.bool_)
#        cnt_colors = np.float32(img[cnt_mask])
#        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#        K = 9
#        ret,label,center=cv.kmeans(cnt_colors,
#                                   K,
#                                   None,
#                                   criteria,
#                                   100,
#                                   cv.KMEANS_PP_CENTERS)
#
#        any_near = False
#        for col in center:
#            if col[0] > 85 and col[0] < 105 and col[1] > 60:# and col[2] > 70:
#                any_near = True
#        if not any_near:
#            continue
#
#        cv.drawContours(yellows, [contour], -1, (0, 255, 0), -1)
#    mask = yellows[:, :, 1] > 1
#    final = image.copy()
#    final[~mask] = [255,255,255]
#    print(sorted(cnts))
#
#    # whites = img.copy()
#    # whites[(whites[:, :, 1] > 30)] = [0,0,0]
#
#    cv.imshow("yellows", yellows) #cv.cvtColor(yellows, cv.COLOR_HSV2RGB))
#    # cv.imshow("whites", cv.cvtColor(whites, cv.COLOR_HSV2RGB))
#    cv.imshow("final_ yellows", final) #cv.cvtColor(yellows, cv.COLOR_HSV2RGB))
#    cv.waitKey(0)
#    return




#    img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
#    img =cv.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
#    img = cv.adaptiveThreshold(img,
#                               255,
#                               cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                               cv.THRESH_BINARY,
#                               15,
#                               5)
#
#    kernel = np.ones((3,3),np.uint8)
#    print(f"{img.mean():.1f}")
#    img = cv.dilate(img, kernel)
#    print(f"{img.mean():.1f}")
#    cv.imshow("Thresholded", img)
#
#
#    # contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#    # cv.drawContours(image, contours, -1, (0, 255, 0), 3)
#    # cv.imshow("With contours", image)
#    cv.waitKey(0)



def main():
    for fp in DIR.glob('*.jpg'):
        print(fp)
        img = cv.imread(str(fp))
        if img.shape[0] < 200 or img.shape[1] < 200:
            continue

        img = segment_colors(img, 17)
        mean_black = np.mean(img[img[:, :, 2] < 160])
        print(mean_black)
        window_name = f"Lightness slider"
        cv.imshow(window_name, img)

        hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        def on_change(val):
            new_img = hsv_img.copy()
            new_img[new_img[:, :, 2] < val] = [0, 0, 0]
            cv.putText(new_img,
                       str(val),
                       (0, new_img.shape[0] - 10),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0,
                       (255, 255, 255),
                       4)
            cv.imshow(window_name, cv.cvtColor(new_img, cv.COLOR_HSV2RGB))

        cv.createTrackbar("lightness cutoff", window_name, 0, 255, on_change)
        cv.waitKey(0)
        continue


        f, a = plt.subplots(figsize=(16, 9))
        tot_img_pixels = img.shape[0] * img.shape[1]

        x_labels = []
        y_values = []

        cv.imshow(f"Orig", img)
        for i in range(2, 20):
            new_img = segment_colors(img, i)
            counts = Counter(tuple(i) for i in new_img.reshape((-1, 3)))
            x_labels.append(i)
            y_values.append(np.std([i/tot_img_pixels for i in counts.values()]))

            if i < 10:
                cv.imshow(f"Segmented {i}", new_img)

        a.plot(x_labels, y_values)
        n_clust = np.array(x_labels)[np.array(y_values) < 0.1][0]
        a.axhline(0.1, color='#ebebeb', ls='--')
        tot_std = np.std([j/(img.shape[0]*img.shape[1]) for j in Counter(tuple(i) for i in img.reshape((-1, 3))).values()])
        print(tot_std)
        a.text(10, np.max(y_values) * 0.8, f"Num clusters: {n_clust} ({tot_std:.6f})")
        plt.show()

        cv.waitKey(0)


        #get_number_plate_box(img)






if __name__ == '__main__':
    main()
