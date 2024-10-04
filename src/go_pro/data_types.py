import av
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import datetime

from dataclasses import dataclass

from src.data_types import Image


@dataclass
class RawImg360:
    frame_number: int
    datetime: datetime
    video_time: float
    frame_iter: int
    metadata: dict

    # From stream 1
    rear_left: Image
    front_left: Image
    front: Image
    front_right: Image
    rear_right: Image

    # From stream 2
    front_top: Image
    rear_top: Image
    rear: Image
    rear_bottom: Image
    front_bottom: Image

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
        img = Image(frame_fronts.to_image())

        self.rear_left = img[:, :688]
        self.front_left = img[:, 688:1376]
        self.front = img[:, 1376:2720]
        self.front_right = img[:, 2720:-688]
        self.rear_right = img[:, -688:]

        # We want to rotate 90 deg and flip
        img = Image(frame_rears).T[::-1]

        self.front_top = img[:688]
        self.rear_top = img[688:1376]
        self.rear = img[1376:2720]
        self.rear_bottom = img[2720:-688]
        self.front_bottom = img[-688:]

    def show_all(self):
        """Plot all images on 2 axes"""
        f, ((a1, a2, a3), (a4, a5, a6)) = plt.subplots(2, 3)
        f.suptitle("Front")
        a1.imshow(self.front_top)
        a2.imshow(self.front_left)
        a3.imshow(self.front)
        a4.imshow(self.front_right)
        a5.imshow(self.front_bottom)
        f.tight_layout()

        g, ((a21, a22, a23), (a24, a25, a26)) = plt.subplots(2, 3)
        g.suptitle("Rear")
        a21.imshow(self.rear_top)
        a22.imshow(self.rear_left)
        a23.imshow(self.rear)
        a24.imshow(self.rear_right)
        a25.imshow(self.rear_bottom)
        g.tight_layout()
        plt.show()


class Img360:
    boundary: int = 32

    raw: RawImg360 # Raw image

    # Stitched frames
    front: Image
    rear: Image
    right: Image
    left: Image
    top: Image
    bottom: Image

    def __init__(self,
                 frame_fronts: av.video.frame.VideoFrame,
                 frame_rears: av.video.frame.VideoFrame):
        """First extract the raw frames and store as a RawImg360, then stitch these raw frames together to make a seamless cube"""
        self.raw = RawImg360(frame_fronts, frame_rears)

        self.front = self.raw.front
        self.rear = self.raw.rear

        self.top = self.stitch_top()
        self.bottom = self.stitch_bottom()
        self.right = self.stitch_right()
        self.left = self.stitch_left()

    def stitch_top(self):
        top = np.vstack(
                (self.raw.front_top[:-self.boundary],
                 self.raw.rear_top[self.boundary:])
        )
        return Image(top)

    def stitch_bottom(self):
        bottom = np.vstack(
                (self.raw.rear_bottom[:-self.boundary],
                 self.raw.front_bottom[self.boundary:])
        )
        return Image(bottom)

    def stitch_right(self):
        right = np.hstack(
                (self.raw.front_right[:, :-self.boundary],
                 self.raw.rear_right[:, self.boundary:])
        )
        return Image(right)

    def stitch_left(self):
        left = np.hstack(
                (self.raw.rear_left[:, :-self.boundary],
                 self.raw.front_left[:, self.boundary:])
        )
        return Image(left)

    @property
    def frame_iter(self) -> int:
        return self.raw.frame_iter

    @frame_iter.setter
    def frame_iter(self, frame_iter: int):
        self.raw.frame_iter = frame_iter

    @property
    def datetime(self) -> int:
        return self.raw.datetime

    @datetime.setter
    def datetime(self, datetime: int):
        self.raw.datetime = datetime

    @property
    def metadata(self) -> int:
        return self.raw.metadata

    @metadata.setter
    def metadata(self, metadata: int):
        self.raw.metadata = metadata

    @property
    def video_time(self) -> int:
        return self.raw.video_time

    @video_time.setter
    def video_time(self, video_time: int):
        self.raw.video_time = video_time

    @property
    def frame_number(self) -> int:
        return self.raw.frame_number

    @frame_number.setter
    def frame_number(self, frame_number: int):
        self.raw.frame_number = frame_number

    def show_all(self):
        """Show all 6 faces of the cube"""
        sml, lrg = self.top.shape[:2]
        big_img = np.zeros((sml+lrg+sml,
                            sml+lrg+sml+lrg,
                            3),
                           np.uint8)

        diff = lrg-sml

        big_img[sml:sml+lrg, :lrg] = self.front
        big_img[sml:sml+lrg, lrg:lrg+sml] = self.right

        big_img[sml:sml+lrg, lrg+sml:lrg+sml+lrg] = self.rear
        big_img[sml:sml+lrg, lrg+sml+lrg:lrg+sml+lrg+sml] = self.left

        big_img[-sml:, lrg+sml:lrg+lrg+sml] = self.bottom
        big_img[:sml, lrg+sml:lrg+lrg+sml] = self.top

        Image(big_img).show()

