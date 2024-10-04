import datetime
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import IntegrityError

from src.db import (
    session
)

from src.db.vehicles import (
    Directory,
    Video,
    Image
)


def add_video_row(TableObj,
                  filepath: Path,
                  obs_datetime: datetime.datetime):
    """Create a video file row and handle the creating of a directory"""
    # First, check if the directory already exists.
    directory = persist_directory(filepath)
    with session() as sesh:
        # Now, create a Video instance with the corresponding directory.
        video = Video(filename=filepath.name,
                      directory=directory,
                      observed_at=obs_datetime)
        sesh.add(video)

        # Commit the changes to the database.
        res = sesh.commit()
    return res


def persist_obj(object_to_insert,
                search_vals: list[str]):
    """Search for an object, if it doesn't exist in the DB then insert and return what was inserted. If it does, then return what is in the DB

    Example:
        obj = persist_obj(
            object_to_insert=Video(filename=str(video_file.name),
                                   directory=video_dir,
                                   observed_at=frame_obs_time),
            search_vals=['filename', 'directory']
        )

    Arguments:
        object_to_insert: The SQLAlchemy object (with data) to insert into the DB
        search_vals: Which vals to use to filter the 'current' data

    Returns:
        object of type type(object_to_insert)
    """
    # The object's type is it's parent -root class
    obj_type = type(object_to_insert)
    with session(expire_on_commit=False) as sesh:
        # the params we use to filter
        where_gen = (getattr(obj_type, val)==getattr(object_to_insert, val)
                     for val in search_vals)

        obj = sesh.execute(
            select(obj_type)
            .where(*where_gen)
        ).scalar_one_or_none()

        if obj is None:
            obj = object_to_insert
            sesh.add(obj)

        sesh.commit()
        return obj


def insert_obj(object_to_insert, ignore_conflicts=False):
    """Create an image file row and handle the creation of the directory"""
    with session() as sesh:
        try:
            sesh.add(object_to_insert)
            res = sesh.commit()
        except IntegrityError as ie:
            # Ignore unique constraint failing
            if ignore_conflicts and 'UNIQUE constraint failed:' in str(ie.orig):
                sesh.rollback()
                return
            raise

    return res


def insert_no_conflict(TableObj,
                       conflict_cols,
                       **values):
    """Create a query that will insert data into a table without a conflict"""
    if isinstance(conflict_cols, str):
        conflict_cols = [conflict_cols]

    with session() as sesh:
        stmt = (
                insert(TableObj)
                .values(**values)
                .on_conflict_do_nothing(index_elements=conflict_cols)
        )
        sesh.execute(stmt)
        res = sesh.commit()
    return res
