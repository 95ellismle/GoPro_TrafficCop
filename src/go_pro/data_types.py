import av
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass


@dataclass
class Img360:
    # From stream 1
    rear_left: None
    front_left: None
    front: None
    front_right: None
    rear_right: None

    # From stream 2
    front_top: None
    rear_top: None
    rear: None
    rear_bottom: None
    front_bottom: None

    def __init__(self,
                 frame_fronts: av.video.frame.VideoFrame,
                 frame_rears: av.video.frame.VideoFrame):
        """Slice up the frames into the 6 sides of a cube

        Args:
            frame_fronts: The frame from the first stream of video (wide horizontal one with front facing camera)
            frame_rears: The frame from the second stream of video (long vertical one with rear facing camera)

        Returns:
            Just sets variables
        """
        img = np.array(frame_fronts.to_image())

        self.rear_left = img[:, :688]
        self.front_left = img[:, 688:1376]
        self.front = img[:, 1376:2720]
        self.front_right = img[:, 2720:-688]
        self.rear_right = img[:, -688:]

        # We want to rotate 90 deg and flip
        img = np.swapaxes(
                np.array(frame_rears.to_image()),
                0,
                1
        )[::-1]

        self.front_top = img[:688]
        self.rear_top = img[688:1376]
        self.rear = img[1376:2720]
        self.rear_bottom = img[2720:-688]
        self.front_bottom = img[-688:]

    def show_all_images(self):
        """Plot all images on 2 axes"""
        f, ((a1, a2, a3), (a4, a5, a6)) = plt.subplots(2, 3)
        a1.imshow(self.front_top)
        a2.imshow(self.rear_top)
        a3.imshow(self.rear)
        a4.imshow(self.rear_bottom)
        a5.imshow(self.front_bottom)
        f.tight_layout()

        g, ((a21, a22, a23), (a24, a25, a26)) = plt.subplots(2, 3)
        a21.imshow(self.front_left)
        a22.imshow(self.rear_left)
        a23.imshow(self.front)
        a24.imshow(self.rear_right)
        a25.imshow(self.front_right)
        g.tight_layout()
        plt.show()

