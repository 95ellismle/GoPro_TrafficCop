import cv2 as cv
import numpy as np


def create_color_pallette(img: np.ndarray, colors: list[tuple[int, int, int]]):
    """Will plot colors in even blocks on img and return img"""
    nrows = int(len(colors) ** 0.5)
    ncols = int(np.ceil(len(colors) / nrows))
    img[:,:] = [0,0,0]
    row_width = len(img[0]) // ncols
    row_height = len(img) // nrows
    div = max(nrows, ncols)

    for i, color in enumerate(colors):
        x = i % div
        y = i // div
        img[y*row_height:(y+1)*row_height,
            x*row_width:(x+1)*row_width] = color
    return img


def plot_histogram(img: np.ndarray):
    import matplotlib.pyplot as plt
    all_colors = img.reshape((-1, 3))
    for i in range(3):
        color = ('r', 'g', 'b')[i]
        counts, bins = np.histogram(all_colors[:, i], 256, [0, 256])
        mean = np.mean(all_colors[:, i])
        std = np.std(all_colors[:, i])

        plt.plot(bins[:-1], counts, color=color)
        plt.plot([mean, mean], [0, max(counts)], '--', color='#eaeaea')
        plt.plot([mean-(1.5*std), mean-(1.5*std)],
                 [0, max(counts)],
                 '--',
                 color=color,
                 alpha=0.2)
        plt.plot([mean+(1.5*std), mean+(1.5*std)],
                 [0, max(counts)],
                 '--',
                 color=color,
                 alpha=0.2)

    plt.show()


def get_color_segmentations(img: np.ndarray,
                            N: int,
                            do_median=True,
                            min_clust_size=None):
    """Returns flattened image with colors segmented and the colors that comprise the image
    """
    def _group_clusts(clusts):
        all_clusters = [[] for i in range(N)]
        for i, lab in enumerate(labels):
            all_clusters[lab[0]].append(i)
        return all_clusters

    flattened = img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv.kmeans(flattened,
                                   N,
                                   None,
                                   criteria,
                                   10,
                                   cv.KMEANS_PP_CENTERS)

    # Remove any small clusters
    ok_inds = list(range(N))
    if min_clust_size:
        if 0 < min_clust_size < 1:
            min_clust_size = len(flattened) * min_clust_size
        min_clust_size = int(min_clust_size)

        clust_groups = _group_clusts(labels)
        for i, inds in enumerate(clust_groups):
            if len(inds) < min_clust_size:
                centers[i] = [0, 0, 0]
        ok_inds = [i for i in range(N) if len(clust_groups[i]) >= min_clust_size]

    # Convert means to medians
    if do_median:
        if not min_clust_size:
            clust_groups = _group_clusts(labels)
        for i in ok_inds:
            mean = centers[i]
            all_colors = flattened[clust_groups[i]]
            std = (np.std(all_colors, axis=0) * 1.5) + 1e-5
            all_colors = all_colors[(all_colors >= (mean - std)).all(axis=1) &
                                    (all_colors <= (mean + std)).all(axis=1)]
            centers[i] = np.median(all_colors, axis=0)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res, centers


def segment_colors(img: np.ndarray, N: int):
    """Returns an image with colors reduced to the amount specified by N"""
    img = cv.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
    flattened_img, colors = get_color_segmentations(img, N)
    ret = flattened_img.reshape((img.shape))
    return ret


