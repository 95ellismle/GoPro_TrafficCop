# Notes:
A document for storing notes, learnings & ideas as I go along with the project. It's probs best to date notes and summarise with a title for easy reference.


To make the project manageable I'll restrict the scope to only handling footage from GoPro Max (360 mode) around London.


## 21/01/2024: Starting thoughts, picking out traffic lights.
I'll start with the ML part of this first -as it's the most fun and what I'm least familiar with.

I'd like to get a set of labelled data to pick out various objects in the project. Though probably a fairly easy one to start is cars skipping lights. I'd like to be able to pick out cars from an image and traffic lights. To train my model I'll need to feed in lots of images of both I imagine. There may be resources online but to learn I'd like to do this step myself.

Start with traffic lights as getting labelled data for these is probably going to be much easier.

### Gathering labelled traffic light data:
Be very tolerant of false positives, and very stringent with false negatives. It's better to not restrict the scope of the labelled data to whatever the CV traffic light finder can find -rather get lots and I can manually sift through and remove.

  1) Try converting the colour to HSV colour space
  2) threshold the image using red, amber and green -red higher, amber lower, green bottom. Start by assuming the camera is always vertically aligned with the y-axis.
    a) Once a red light is found, skip ahead some frames and see if it turns amber then green. If it does assume it's a traffic light.
  3) Using contours, erosion, dilation to detect a box surrounding all three lights.
  4) Print out frames of traffic lights as a series of JPG and I can cycle through and delete any bad ones manually.


## 21/01/2024: Format of the GoPro 360 video file
The 360 video file contains 6 streams of data. These are:
   1) video track (front horizontal strip)
   2) video track (rear vertical strip)
   3) audio track
   4) audio track? PCM -pulse code modulation.
   5) telemetry/metadata track
   6) timecode track. This might be timecodes for data measurements.

The python library `gpmf` and `ffmpeg-python` can be used to extract the various streams from the video file.

We have front video, back video and various quantities from the video like speed, lat/lon, wetness, acceleration, magnetic field strength & tilt.

This is useful for docs: https://github.com/gopro/gpmf-parser


# 03/02/2024: Format of individual GoPro frames
We have 2 streams of video, front-ish and rear-ish. The full spherical view of the 3D world is split into 10 images:
front, front-left, front-right, rear-left, rear-right
rear, rear-top, rear-bottom, front-top, front-bottom


*front*
The front-ish frames come in a strip that wraps horizontally around you.
Imagine you're wearing cyclops goggles that block your view upwards and downwards.

E.g: in the graphic below imagine you are the dot and we are looking downwards at your head.
     The lines around you is what the first stream captures.
                /-----\
                \  .  /

*rear*
The rear-ish frames come in a strip that wraps vertically around you. It fills the strip left over from the first stream (see front).
Imagine you are lying on your side and you are facing backwards with cyclops goggles on (as in the front strip). The strip you see
running top to bottom is what is captured.

E.g: in the graphic below imagine you are the line, lying on your side and the cross is your head.
     The vertical line shows the strip you can see -it wraps in a C shape vertically.

       |
       +-----
       |

# 04/02/2024: Picking out traffic lights
1) Find red, yellow green circles. This can be done with HSV filtering, erosion/dilation to clean up and findContours. Compare area of contour to area of the minEnclosingCircle.

2) Find corner features near each light. Probably first threshold black-ish objects (the traffic light box). Can also use FLANN: https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html to find lights (though they may be rotated etc...).


# 24/09/2024: Data Track
The data track can be parsed with the `gpmf` library. If, upon importing, an 'probe not found' error is raised: do `pip uninstall python-ffmpeg` then `pip install ffmpeg-python`.

The data track can be read via:

```
filepath: str = "/Users/mattellis/Downloads/GSX.mp4"
stream = gpmf.io.extract_gpmf_stream(filepath)
```

To extract all the data within the track then the following can be used:
```
expanded_stream = gpmf.parse.expand_klv(stream)
```

Rough structure of `expanded_stream`:
```
[   # List of second snapshots (each index shows data for that second, e.g: 0 = data for 0->1 seconds)
    [
        # Metadata
        "DEVC",
        <KLVLength object -length of stream of data in frame>,

        [
            # Actual Data
0           'DVID',  # Device ID
1           'DVNM',  # Device Name
2           'STRM',  # Acceleration
3           'STRM',  # Gyroscope
4           'STRM',  # Magnetometer
5           'STRM',  # Shutter speed (exposure time)
6           'STRM',  # White balance (in Kelvin)
7           'STRM',  # White balance (in RGB Gain)
8           'STRM',  # Sensor ISO
9           'STRM',  # Image uniformity
10          'STRM',  # GPS (Lat., Long., Alt., 2D speed, 3D speed)
11          'STRM',  # Camera Orientation
12          'STRM',  # Image Orientation
13          'STRM',  # Disparity track
14          'STRM',  # Gravity Vector
15          'STRM',  # Wind processing
16          'STRM',  # Mic wet
17          'STRM',  # Audio level
        ]
]
```


# 03/10/2024: System design
Use a microservices architecture.

To fill out data in a main `Vehicle` table. Only need to fill out the skeleton data (trivial data to retrieve e.g: observed_at, img_filepath, video_filepath, video_time, location).

To fill in other columns have services that read this table and fill in any columns that are NULL.

A similar idea can be used for incidents and other reporting. For example: say there are some rows in the Vehicle table that shows untaxed vehicles driving around, then incidents can be created for these and saved in the database. A separate service can be used to query the incident table to create email reports for me to check.
