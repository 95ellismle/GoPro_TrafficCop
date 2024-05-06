from pathlib import Path

import click

from src.ml_detect import Detect
from src.go_pro.utils import read_360_video




def main(video_file):
    for frame in read_360_video(video_file, min_time=1360*90_000):
        print("\r"f"Frame: {frame.frame_number}       ", end="\r")
        extractor = Detect(frame.front)
        frame.front.show()
        for car in extractor.extract('car'):
            pass
        import ipdb; ipdb.set_trace()



@click.command()
@click.argument('video_file', type=click.Path(exists=True))
def main_cli(video_file: Path):
    """Analyse a video file and pick out cars that skip red lights"""
    video_file = Path(video_file)
    main(video_file)


if __name__ == '__main__':
    main_cli()
