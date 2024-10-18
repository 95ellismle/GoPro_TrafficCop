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


class Img360_Cube:
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

    def _add_extra_bits(self,
                       main, left, right, top, bottom,
                       extra=0.2):
        """Add extra bits to the frame from surrounding frames


                             top
                              |
                     left -  main  - right
                              |
                            bottom


        e.g: If we want to add some extra bits to the rear image, we'd add:
            main = rear frame
            left = right frame
            right = left frame
            top = top frame
            bottom = bottom frame
        """
        if extra > 1: extra = 1
        left = left[:,-int(left.shape[1] * extra):]
        right = right[:,:int(right.shape[1] * extra)]
        top = top[-int(top.shape[0] * extra):]
        bottom = bottom[:int(bottom.shape[0] * extra)]

        new_img = np.vstack((top, main, bottom))

        left_wing = np.zeros((new_img.shape[0], left.shape[1], 3), np.uint8)
        left_wing[top.shape[0]:top.shape[0]+left.shape[0]] = left

        right_wing = np.zeros((new_img.shape[0], right.shape[1], 3), np.uint8)
        right_wing[-bottom.shape[0]-right.shape[0]:-bottom.shape[0]] = right

        new_img = np.hstack((left_wing, new_img, right_wing))

        new_img = Image(new_img)

        return new_img

    def get_front_extra(self, extra=0.2):
        return self._add_extra_bits(
                main=self.front,
                left=self.left,
                right=self.right,
                bottom=np.rot90(self.bottom, 2),
                top=np.rot90(self.top, 2),
                extra=extra)

    def get_rear_extra(self, extra=0.2):
        return self._add_extra_bits(
                main=self.rear,
                left=self.right,
                right=self.left,
                bottom=self.bottom,
                top=self.top,
                extra=extra)

    def get_left_extra(self, extra=0.2):
        return self._add_extra_bits(
                main=self.left,
                left=self.rear,
                right=self.front,
                bottom=np.rot90(self.bottom, 1),
                top=np.rot90(self.top, 3),
                extra=extra)

    def get_right_extra(self, extra=0.2):
        return self._add_extra_bits(
                main=self.right,
                left=self.front,
                right=self.rear,
                bottom=np.rot90(self.bottom, 3),
                top=np.rot90(self.top, 1),
                extra=extra)

    def get_top_extra(self, extra=0.2):
        return self._add_extra_bits(
                main=self.top,
                left=np.rot90(self.right, 3),
                right=np.rot90(self.left, 1),
                bottom=self.rear,
                top=np.rot90(self.front, 2),
                extra=extra)

    def get_bottom_extra(self, extra=0.2):
        return self._add_extra_bits(
                main=self.bottom,
                left=np.rot90(self.right, 1),
                right=np.rot90(self.left, 3),
                bottom=np.rot90(self.front, 2),
                top=self.rear,
                extra=extra)

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


def _normalise_arr(arr: np.ndarray, norm_val: float = 1.0):
    """Normalise an array of values to fit in the range [0, norm_val]"""
    min_, max_ = arr.min(), arr.max()
    new_norm = norm_val / (max_ - min_)
    arr = (arr - min_) * new_norm
    return arr

def _create_perspective_uv(total_dist: float,
                           depth: float,
                           dx: np.ndarray,
                           dy: np.ndarray,
                           dz: np.ndarray,
                           new_height: float = 1,
                           new_width: float = 1,
    ):
    """Creates u & v coords from the params

        u =  dx * t
            --------
             z + dy

        v =  dz * t
            --------
             z + dy

    Note:
        All units are relative to h/2 (half width of main image)

    Args:
        total_dist: Distance from light source to screen
        depth: Distance from light source to nearest edge of cubenet
        dx, dy, dz: The x, y and z coordinates
        new_height: Height of the returned img coords
        new_width: Width of the returned img coords
        additional_factor: A number to multiply the values by

    Returns:
        u, v arrays which point to the place each coord should go in the image to
        achieve the required perspective.
    """
    u_shape = dy.shape[1] if isinstance(dy, np.ndarray) else dx.shape[1]
    v_shape = dz.shape[0] if isinstance(dz, np.ndarray) else dy.shape[0]
    factor = total_dist / (depth + dy)
    u = dx * factor
    v = dz * factor
    return _normalise_arr(u, new_width), _normalise_arr(v, new_height)


def linear_interpolation(image: np.ndarray) -> np.ndarray:
    """Fill in black gaps with linear interpolation of pixels beside them"""
    new_img = image.copy()

    # First get the gaps -only get the middle ones as edges are awkward with indexing
    h, w = image.shape[:2]
    coords = np.stack(np.meshgrid(np.arange(h-2), np.arange(w-2), indexing='ij'), axis=-1)
    black_spots = coords[(image[1:-1,1:-1] == [0,0,0]).all(axis=2)] + 1

    # Iterate over the black spots in chunks and interpolate
    chunksize = 2_000_000
    for i in range(0, len(black_spots), chunksize):
        points_to_interp = black_spots[i:i+chunksize]

        new_points = new_img[points_to_interp[:,0], points_to_interp[:,1]].astype(np.float64)
        counts = np.zeros(len(new_points))
        for x_shift in range(-1, 2):
            for y_shift in range(-1, 2):
                if x_shift == 0 and y_shift == 0: continue

                fact = abs(x_shift) + abs(y_shift)
                pts = image[points_to_interp[:,0] + x_shift, points_to_interp[:, 1] + y_shift] / fact
                msk = (pts > 0).any(axis=1)
                counts[msk] += 1/fact
                new_points[msk] += pts[msk]

        with np.errstate(divide='ignore', invalid='ignore'):
            new_img[points_to_interp[:,0], points_to_interp[:,1]] = (new_points / counts[:,None]
                                                                     ).round().astype(np.uint8)

    return new_img


def _get_perspective_view_cubenet_faces(
        left, main, right, top, bottom,
        depth, img_dist):
    """A helper for the `get_perspective_view_cubenet` see docstr for that.

    Args:
        left: Face to the left of the main one
        main: Main face to project
        right: Face to the right of the main one
        top: Face to the top of the main one
        bottom: Face to the bottom of the main one
        depth: Distance from light source to nearest edge of box
        img_dist: Distance from nearest edge of box to the image plane
    """
    assert main.shape[0] == main.shape[1], f"Main image needs to be square"
    total_dist = depth + 1 + img_dist
    h_prime = int(main.shape[0] // 2)
    uv = {}

    # Left
    y_spacer = np.linspace(0, 1, left.shape[1], dtype=np.float32)
    z_spacer = np.linspace(-1, 1, left.shape[0], dtype=np.float32)
    dy, dz = np.meshgrid(y_spacer, z_spacer)
    new_height = (2/depth) + 2
    u, v = _create_perspective_uv(total_dist, depth, dx=-1, dy=dy, dz=dz, new_height=new_height, new_width=1)

    # Make the right edge of the left trapezoid the same size as the front img
    scaler = main.shape[0] / (v[-1][-1] - v[0][-1])
    v *= scaler
    u *= scaler
    v = v.round().astype(int)
    u = u.round().astype(int)

    persp = np.zeros((1 + v[-1, 0] - v[0,0], 1 + u[0, -1] - u[0, 0], 3), np.uint8)
    persp[v.flatten(), u.flatten()] = left.arr.reshape((-1, 3))
    persp = linear_interpolation(persp)
    uv['left'] = {'u': u, 'v': v, 'persp_img': persp}

    # Just slap teh main image in the middle
    new_main = np.zeros((persp.shape[0], main.shape[1], 3), np.uint8)
    y_base = int((persp.shape[0] - main.shape[0]) / 2)
    new_main[y_base:y_base+main.shape[1],:] = main
    Image(np.hstack((persp, new_main))).show()
    import ipdb; ipdb.set_trace()
    uv['main'] = {'persp_img': new_main}

    # Right
    y_spacer = np.linspace(0, 1, left.shape[1], dtype=np.float32)
    z_spacer = np.linspace(-1, 1, left.shape[0], dtype=np.float32)
    dy, dz = np.meshgrid(y_spacer, z_spacer)
    u, v = _create_perspective_uv(total_dist, depth, dx=-1, dy=dy, dz=dz)
    import ipdb; ipdb.set_trace()
    persp = np.zeros_like(right.arr)
    persp[(v.flatten()-0.5).round().astype(int), (u.flatten()-0.5).round().astype(int)] = right.arr.reshape((-1, 3))
    uv['right'] = {'u': u, 'v': v, 'persp_img': persp}


    Image(np.hstack((left.arr, main.arr, right.arr))).show()
    Image(np.hstack((uv['left']['persp_img'],
                     uv['main']['persp_img'],
                     uv['right']['persp_img'],))
    ).show()

    import ipdb; ipdb.set_trace()
    import matplotlib.pyplot as plt
    import ipdb; ipdb.set_trace()
    plt.imshow(u); plt.show()
    o=1



def get_perspective_view_cubenet(
        img_cube: Img360_Cube,
        depth: float = 2.7,
        img_dist: float = 3.0,
        face: str = "front",
        factor: float = 0.5):
    """Will get a perspective view of the cubenet, i.e. as if a light source were
    placed behind it and illuminated the folded cube onto a flat plane img_dist away.

    Units are relative to the length of half a cube edge length.


        img_dist
          <->
           /|
          / |
   depth /  |
   <--> /   |
       /    |
      /|--| |
     / |  | |
    .  |  | |
     \ |  | |
      \|--| |
       \    |
        \   |
         \  |
          \ |
           \|
    Args:
        img_cube: The Img360_Cube object with a cube net of the img
        depth: The distance away from the cube net to place the light source
        img_dist: The distance from the nearest edge of the cube net to place the persepective view
        factor: What percentage of the surrounding faces we project

    Returns:
        Image, The view of the cubenet projected onto a 2D flat image as if a light source were place behind it.
    """
    match face:
        case "front":
            img = _get_perspective_view_cubenet_faces(left=img_cube.left[:, -int(img_cube.left.shape[1]*factor):],
                                                      main=img_cube.front,
                                                      right=img_cube.right[:, :int(img_cube.right.shape[1]*factor)],
                                                      top=Image(np.rot90(img_cube.top[:int(img_cube.top.shape[0]*factor)], 2)),
                                                      bottom=Image(np.rot90(img_cube.bottom[-int(img_cube.bottom.shape[0]*factor):], 2)),
                                                      depth=depth,
                                                      img_dist=img_dist)
        case _:
            raise ValueError(f"Don't recognise {face}")
    img.show()
    raise SystemExit


class Img360_Sphere:
    """Probs not needed -never finished"""
    raw_cube_net: Img360_Cube

    def __init__(self, cube_img: Img360_Cube):
        self.raw_cube_net = cube_img
        self.front_hemisphere = self._get_front_hemisphere()
        self.rear_hemisphere = self._get_rear_hemisphere()

    def _get_front_hemisphere(self):
        cube = self.raw_cube_net
        hemisphere = {
            'left': cube.left[:, -cube.left.shape[1]//2:],
            'front': cube.front,
            'right': cube.right[:, :cube.right.shape[1]//2],
            'top': Image(np.rot90(cube.top[:cube.top.shape[0]//2], 2)),
            'bottom': Image(np.rot90(cube.bottom[-cube.bottom.shape[0]//2:], 2)),
        }
        lat_lon = {}

        # Front
        z_space = np.linspace(-1, 1, hemisphere['front'].shape[0])
        x_space = np.linspace(1, -1, hemisphere['front'].shape[1])
        x, z = np.meshgrid(x_space, z_space)
        r = np.sqrt(x**2 + 1 + z**2)
        lat_lon['front'] = {
                'r': r,
                'theta': np.arccos(z / r),
                'phi': np.arctan(-1 / x)}
        lat_lon['front']['phi'][:,:len(lat_lon['front']['phi'])//2] += np.pi

        # Left
        z_space = np.linspace(-1, 1, hemisphere['left'].shape[0])
        y_space = np.linspace(0, -1, hemisphere['left'].shape[1])
        z, y = np.meshgrid(z_space, y_space)
        z, y = z.T, y.T
        r = np.sqrt(1 + y**2 + z**2)
        lat_lon['left'] = {
                'r': r,
                'theta': np.arccos(z / r),
                'phi': np.pi + np.arctan(y)}

        # Right
        z_space = np.linspace(1, -1, hemisphere['right'].shape[0])
        y_space = np.linspace(-1, 0, hemisphere['right'].shape[1])
        z, y = np.meshgrid(z_space, y_space)
        z, y = np.rot90(z, 1), np.rot90(y, 1)
        r = np.sqrt(1 + y**2 + z**2)
        lat_lon['right'] = {
                'r': r,
                'theta': np.arccos(z / r),
                'phi': np.arctan(-y)}

        # Top
        x_space = np.linspace(1, -1, hemisphere['top'].shape[0])
        y_space = np.linspace(-1, 0, hemisphere['top'].shape[1])
        x, y = np.meshgrid(z_space, y_space)
        r = np.sqrt(x**2 + y**2 + 1)
        lat_lon['top'] = {
                'r': r,
                'theta': np.arccos(1 / r),
                'phi': np.arctan(y/x)}
        lat_lon['top']['phi'][:,:len(lat_lon['top']['phi'])//2] += np.pi

        # Bottom
        x_space = np.linspace(1, -1, hemisphere['bottom'].shape[0])
        y_space = np.linspace(0, -1, hemisphere['bottom'].shape[1])
        x, y = np.meshgrid(z_space, y_space)
        r = np.sqrt(x**2 + y**2 + 1)
        lat_lon['bottom'] = {
                'r': r,
                'theta': np.arccos(-1 / r),
                'phi': np.arctan(y/x)}
        lat_lon['bottom']['phi'][:,:len(lat_lon['bottom']['phi'])//2] += np.pi

        name = 'top'
        u = lat_lon[name]['r']*np.sin(lat_lon[name]['theta']) * np.cos(lat_lon[name]['phi'])
        u -= u.min()
        u /= u.max()
        u *= hemisphere[name].shape[1]
        v = lat_lon[name]['r']*np.sin(lat_lon[name]['theta']) * np.sin(lat_lon[name]['phi'])
        v -= v.min()
        v /= v.max()
        v *= hemisphere[name].shape[0]

        Image(cv.remap(hemisphere[name].arr, u.astype(np.float32), v.astype(np.float32), interpolation=cv.INTER_CUBIC)).show()
        import ipdb; ipdb.set_trace()

        # plt.imshow(np.vstack((lat_lon['bottom']['theta'], lat_lon['front']['theta'], lat_lon['top']['theta']))); plt.show()
        # plt.imshow(np.vstack((lat_lon['bottom']['r'], lat_lon['front']['r'], lat_lon['top']['r']))); plt.show()
        # f = plt.figure()
        # plt.imshow(np.vstack((lat_lon['bottom']['phi'], lat_lon['front']['phi'], lat_lon['top']['phi'])))
        # f.suptitle("Top to Down")
        # plt.show()

        # plt.imshow(np.hstack((lat_lon['left']['theta'], lat_lon['front']['theta'], lat_lon['right']['theta']))); plt.show()
        # plt.imshow(np.hstack((lat_lon['left']['r'], lat_lon['front']['r'], lat_lon['right']['r']))); plt.show()
        # f = plt.figure()
        # plt.imshow(np.hstack((lat_lon['left']['phi'], lat_lon['front']['phi'], lat_lon['right']['phi'])))
        # f.suptitle("Left to Right")
        # plt.show()

        raise SystemExit("EXIT BEFORE RETURN OF _get_front_hemisphere")
        return hemisphere

    def _get_rear_hemisphere(self):
        rear_hemisphere = None
        return rear_hemisphere
