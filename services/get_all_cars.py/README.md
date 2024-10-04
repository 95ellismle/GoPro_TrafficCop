# Purpose
This service will run through each frame in a video straight from the GoPro and save all the images of cars to disk.
It will also populate the following items in the DB:
    * Location.@
    * Directory.@
    * Video.@
    * Image.@
    * Vehicle.video_id
    * Vehicle.location_id
    * Vehicle.vehicle_image_id
    * Vehicle.observed_at
    * Vehicle.time
    * Vehicle.created_at

# How to run
To run this service there are 2 options:
    * video_file [Path] Path to the video file to analyse
    * --output-dir [Path] Path to the place to save the outputted images of cars (will append f"{video_file.name}-{datetime_of_video}")

Example:
    python services/get_all_cars.py/main.py ~/Downloads/saahf_laandan.mp4 --output-dir /tmp/frames


