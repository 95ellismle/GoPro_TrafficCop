from pathlib import Path

import click

from src.ml_detect.cars import Detect
from src.go_pro.utils import read_360_video




def main(video_file, start_frame):
    cont = False
    for frame in read_360_video(video_file, start_frame):
        if cont is False:
            cont = True
            continue
        print("\r"f"Frame: {frame.frame_number}       ", end="\r")
        extractor = Detect(frame.front)
        frame.front.show()
        for car in extractor.extract('car'):
            pass
        import ipdb; ipdb.set_trace()



@click.command()
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--start_frame', type=int, default=0)
def main_cli(video_file: Path, start_frame: int):
    """Analyse a video file and pick out cars that skip red lights"""
    video_file = Path(video_file)
    main(video_file, start_frame)


if __name__ == '__main__':
    main_cli()
