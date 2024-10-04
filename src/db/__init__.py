from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from db.vehicles import (
    VehicleBase,
    NumberPlate,
    VehicleMake,
    Directory,
    Video,
    Image,
    Color,
    Location,
    VehicleType,
    Vehicle,
)

from src.settings import sql_path

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(sql_path, echo=True)
    return _engine


def create_all_tables():
    engine = get_engine()
    VehicleBase.metadata.create_all(_engine)


@contextmanager
def session(*args, **kwargs):
    engine = get_engine()
    with Session(engine, *args, **kwargs) as session:
        yield session


def insert_no_conflict(TableObj,
                       conflict_col,
                       **values):
    query = insert_no_conflict_create_query(
            TableObj,
            conflict_col=conflict_col,
            **values)
    with session() as sess:
        res = sess.execute(query)
        sess.commit()
    return res
