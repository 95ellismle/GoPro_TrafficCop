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

