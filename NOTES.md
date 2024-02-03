# Notes:
A document for storing notes, learnings & ideas as I go along with the project. It's probs best to date notes and summarise with a title for easy reference.


To make the project manageable I'll restrict the scope to only handling footage from GoPro Max (360 mode) around London.


## 21/01/2024: Starting thoughts, picking out traffic lights.
I'll start with the ML part of this first -as it's the most fun and what I'm least familiar with.

I'd like to get a set of labelled data to pick out various objects in the project. Though probably a fairly easy one to start is cars skipping lights. I'd like to be able to pick out cars from an image and traffic lights. To train my model I'll need to feed in lots of images of both I imagine. There may be resources online but to learn I'd like to do this step myself.

Start with traffic lights as getting labelled data for these is probably going to be much easier.

### Gathering labelled traffic light data:
Be very tolerant of false positives, and very stringent with false negatives. It's better to not restrict the scope of the labelled data to whatever the CV traffic light finder can find -rather get lots and I can manually sift through and remove.

  1) threshold the image using red, amber and green -red higher, amber lower, green bottom. Start by assuming the camera is always vertically aligned with the y-axis.
    a) Once a red light is found, skip ahead some frames and see if it turns amber then green. If it does assume it's a traffic light.
  2) Using contours, erosion, dilation to detect a box surrounding all three lights.
  3) Print out frames of traffic lights as a series of JPG and I can cycle through and delete any bad ones manually.


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
