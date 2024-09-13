import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from src.data_types import (
    Circle,
    Color,
    Combine,
    HSV,
    Image,
    MinMax,
)


ZERO_SHAPE_CONT = np.array([[[0, 0]], [[0, 6]], [[10, 6]], [[10, 0]]], dtype=np.int32)


def snip_little_lines(img: np.ndarray,
                      snip_size: tuple[int, int] = (2, 2)):
    """Will perform a morphological open to cut away fine lines in a black and white imagw

    Args:
        img: black and white image
        snip_size: How big a snip to perform on images
    """
    kernel = np.ones(snip_size, np.uint8)
    snip_func = lambda i: cv.erode(i, kernel)

    # If odd then don't shift, else shift by -1,-1
    if snip_size[0] % 2 == 1 and snip_size[1] % 2 == 1:
        return snip_func(img)

    else:
        shifter = (snip_size[0] % 2, snip_size[1] % 2)
        new_img = np.zeros((img.shape[0]+shifter[0], img.shape[1]+shifter[1]),
                           np.uint8)
        new_img[:img.shape[0], :img.shape[1]] = img
        new_img = snip_func(new_img)
        return new_img[shifter[0]:, shifter[1]:]


def rectangliness(img: np.ndarray):
    """Get a score for how rectangular a contour is

    Args:
        img: A contour from findContours
    """
    import ipdb; ipdb.set_trace()
    o=1


def derivative_snipping(img: np.ndarray,
                        threshold: float = 0.01,
                        max_col_edge: float = 0.2,
                        max_row_edge: float = 0.1):
    """For a number of columns and rows and if there are any sections of white
    lasting more than a threshold then fill them in.

    Outline of algorithm:
        1) get all rows/columns within a certain distance from the edge (default is 20%)
        2) for each row/col iterate across the line with 2 pointers, if there is a chain of white pixels less than the (threshold * len of line) across: black them out.

    Args:
        img: bw, thresholded image. 2D array of either 0 or 255
        threshold: max percentage of consequtive white pixels
        max_col_edge: How far into each vertical edge to perform the snipping
                        e.g: 0.1 would only do the snipping 10% in from the right & left
        max_row_edge: How far into each horizontal edge to perform the snipping
                        e.g: 0.1 would only do the snipping 10% in from the top & bottom
    """
    o_img = img.copy()
    nrows, ncols = img.shape

    # Do columns
    thresh = round(ncols * threshold)
    if thresh == 0: thresh = 1
    if max_col_edge > 0.5: max_col_edge = 0.5
    col_slice = int(ncols*max_col_edge)
    cols_to_do = list(range(0, col_slice)) + list(range(ncols-col_slice, ncols))
    for i in cols_to_do:
        line = o_img[:, i]
        pt1, pt2 = 0, 1
        bad_inds = []
        while pt2 < len(line):
            if line[pt1] != line[pt2]:
                if pt2 - pt1 <= thresh:
                    bad_inds.extend(list(range(pt1, pt2)))
                pt1 = pt2
            pt2 += 1
        if bad_inds:
            img[bad_inds, i] = 0

    # Do rows
    thresh = round(nrows * threshold)
    if thresh == 0: thresh = 1
    if max_row_edge > 0.5: max_row_edge = 0.5
    row_slice = int(nrows*max_row_edge)
    rows_to_do = list(range(0, row_slice)) + list(range(nrows-row_slice, nrows))
    for i in rows_to_do:
        line = o_img[i, :]
        pt1, pt2 = 0, 1
        bad_inds = []
        while pt2 < len(line):
            if line[pt1] != line[pt2]:
                if pt2 - pt1 <= thresh:
                    bad_inds.extend(list(range(pt1, pt2)))
                pt1 = pt2
            pt2 += 1
        if bad_inds:
            img[i, bad_inds] = 0

    return img


def count_islands(img: np.ndarray,
                  color: None | int = None):
    """Count how many islands in a bw image that are of a certain shade

    Args:
        img: Black-white image
        color: [optional] filter for only these colors
    """
    conts, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    import ipdb; ipdb.set_trace()




def flood_fill_edges(img: np.ndarray,
                     fill_col: int = 0,
                     search_col: int = 255,
                     center_tol: float = 0.9,
                     max_change: int | None = None):
    """Will flood fill only the edges of a grayscale image with a value

    By default we color in white edges in black
    """
    m_img = img.copy()
    m_img[m_img < 100] = 0
    m_img[m_img >= 100] = 255
    h, w = m_img.shape

    # Original color of the center of the image -this shouldn't change
    dlt = (h - 8) / (2 * h)
    dlt = max(min((0.4, dlt)), 0)
    orig_mean = m_img[int(h*dlt):int(h*(1-dlt)),int(w*dlt):int(w*(1-dlt))].mean()

    # Loop over the 4 sides
    for x, y in ((0,...),
                 (...,0),
                 (len(m_img)-1,...),
                 (...,len(m_img[0])-1)):

        # Get coordinate of the white patches on 1 edge
        inds = np.argwhere(m_img[x, y] == search_col)
        if len(inds) == 0: continue
        inds = (inds[[True] + list(np.diff(inds[:, 0]) != 1)])[:,0]

        # Loop over each section that might need filling
        for next_ind in inds:
            coords = (y, next_ind)
            if y is ...:
                coords = (next_ind, x)

            orig_img = m_img.copy()
            m_img = cv.floodFill(m_img, np.zeros((h+2,w+2),np.uint8), coords, fill_col)[1]

            # We don't want to be flood filling the center
            # Just undo...
            new_mean = m_img[int(h*dlt):int(h*(1-dlt)),int(w*dlt):int(w*(1-dlt))].mean()
            if (abs(new_mean - orig_mean) / orig_mean) >  (1-center_tol):
                m_img = orig_img
                continue

            # If the image has changed color more than max_change, then undo operation
            if max_change is None: continue
            if m_img.sum() < (max_change * orig_img.sum()):
                m_img = orig_img
    return m_img


def get_contour_mask(img: np.ndarray,
                     contour: np.ndarray,
                     contour_fill: int=-1):
    """Will create a mask and get the pixel points"""
    mask = np.zeros_like(img)
    cv.drawContours(mask, [contour], -1, 1, contour_fill)
    mask = mask.astype(np.bool_)
    return mask


def remove_small_contours(img: np.ndarray,
                          threshold: float = 5e-4,
                          contour_fill_color: int = -1):
    """Will remove any small contours from binary image

    Args:
        threshold = percent of the image's total area the contour can take
    """
    img = img.copy()
    img[img < 200] = 0
    img[img >= 200] = 255
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    bad_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area < 6:
            bad_contours.append(contour)
            continue

        pct_area = area / img_area
        if pct_area < threshold:
            bad_contours.append(contour)

    if len(bad_contours):
        bad_contours.sort(key=lambda i: cv.contourArea(i), reverse=True)
        for contour in bad_contours:
            # Get the outline color of the contour and fill with that
            if contour_fill_color == -1:
                mask_full = np.uint8(get_contour_mask(img, contour, -1)) * 255
                mask_overfull = cv.dilate(mask_full, np.ones((5,5),np.uint8))
                mask_outer = np.bool_(mask_overfull ^ mask_full)

                counts = {v: k for k, v in Counter(img[mask_outer]).items()}
                color = counts[max(counts)]
            else:
                color = contour_fill_color

            cv.drawContours(img, [contour], -1, int(color), -1)

    return img


def join_clean_mask(img: np.ndarray):
    """Will try to join up contours in a mask binary image and remove the bad looking ones.

    This is to remove smallish, black speckles in a binary image.
    """
    img = img.copy()
    img[img > 180] = 255
    _, thresh = cv.threshold(img,0,255,cv.THRESH_TOZERO+cv.THRESH_OTSU)
    thresh = cv.adaptiveThreshold(cv.GaussianBlur(thresh, (3,3), 3),
                                  255,
                                  cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY,
                                  15,
                                  2)

    for i in ((2, 2),):
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, np.ones(i, np.uint8))
    thresh = flood_fill_edges(thresh)

    contours, hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    bad_contours = []
    frame_area = img.shape[0] * img.shape[1]
    for contour, (nxt, prv, child1, parent) in zip(contours, hierachy[0]):
        mask = np.zeros_like(img)
        cv.drawContours(mask, [contour], -1, 255, -1)
        mean_col = cv.mean(img, mask=mask)[0]

        area = cv.contourArea(contour)
        if area < 6: bad_contours.append(contour); continue

        pct_area = area / frame_area
        solidity = float(area) / cv.contourArea(cv.convexHull(contour))
        # The big rectangle that surrounds the number plate
        if pct_area > 0.2 and nxt == -1 and solidity > 0.92:
            if parent == -1:
                crds = np.transpose(np.nonzero(mask))[0][::-1]
                h,w = thresh.shape
                thresh = cv.floodFill(thresh, np.zeros((h+2,w+2),np.uint8), crds, 255)[1]
            continue

        # We don't want to draw too much of the mask away
        if pct_area > 0.05: continue
        # We don't want any very light contours (not many black pixels)
        if mean_col > 180:
            circleliness = cv.matchShapes(ZERO_SHAPE_CONT, contour, 1, 0.0)
            if circleliness > 0.3:
                bad_contours.append(contour); continue
        if area < 30: bad_contours.append(contour); continue
        if 0.99 < pct_area < 0.001: bad_contours.append(contour); continue
        if (area < 30 and solidity > 0.9) or solidity < 0.1: bad_contours.append(contour); continue

    cv.drawContours(thresh, bad_contours, -1, 255, -1)
    thresh = cv.erode(thresh, kernel=np.ones((2, 2), np.uint8))
    ret = np.zeros_like(img) + 255
    mask = thresh == 0
    ret[mask] = img[mask]
    return ret


def contrast_boost(img: np.ndarray):
    """Normalize numbers between 0 -> 255 and apply clahe to the image

    N.B:
        Must be a grayscale image
    """
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    img = np.float64(img)

    min_ = img.min()
    if min_ > 1e-5:
        img -= min_

    max_ = img.max()
    if max_ < 249.9999:
        img *= 255 / max_

    img = clahe.apply(np.uint8(img.round()))

    return img


def apply_threshold(img_arr: np.ndarray,
                    thresholds: dict) -> np.ndarray:
    """Will apply a dictionary of thresholds to the image array.

    The keys of the thresholds dict can be:
        Combine.And, Combine.OR, MinMax.MIN, MinMax.MAX, HSV.HUE, HSV.SATURATION, HSV.LIGHTNESS.

    Args:
        img_arr: numpy array with all thresholds
        thresolds: dictionary with thresholds.

    Examples:
        apply_thresholds(img_arr,
                         {
                          Combine.AND: {
                             MinMax.MIN: {
                                 HSV.HUE: 30
                             },
                             MinMax.MAX: {
                                 HSV.HUE: 150
                             }
                          }
                         }
        )
    """
    for key, value in thresholds.items():
        if isinstance(key, (Combine)):
            if key == Combine.AND:
                mask = np.ones(img_arr.shape[:2], dtype=bool)
                for new_threshold in thresholds[key]:
                    mask = mask & apply_threshold(img_arr, new_threshold)
            elif key == Combine.OR:
                mask = np.zeros(img_arr.shape[:2], dtype=bool)
                for new_threshold in thresholds[key]:
                    mask = mask | apply_threshold(img_arr, new_threshold)
            return mask

        if isinstance(key, (MinMax)):
            assert isinstance(value, dict), f"Badly formatted thresholds dict. MinMax must have a dict entry: {value}"
            assert len(value) == 1, f"MinMax can only have 1 entry -if you want to combine multiple selects use 'and' or 'or': {value}"
            if key == MinMax.MIN:
                ind, val = next(iter(thresholds[MinMax.MIN].items()))
                return img_arr[:,:,ind.value] <= val
            elif key == MinMax.MAX:
                ind, val = next(iter(thresholds[MinMax.MAX].items()))
                return img_arr[:,:,ind.value] >= val
            else:
                raise ValueError(f"HSV must come after MinMax: {thresholds}")


