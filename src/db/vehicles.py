import datetime
from typing import Optional

from sqlalchemy import (
    ForeignKey,
    String,
    DateTime,
    UniqueConstraint,
)

from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.sql import func


class VehicleBase(DeclarativeBase):
    inserted_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class NumberPlate(VehicleBase):
    __tablename__ = "number_plate"

    num_plate_id: Mapped[int] = mapped_column(primary_key=True)
    characters: Mapped[str] = mapped_column(String(7), unique=True)


class VehicleMake(VehicleBase):
    __tablename__ = "vehicle_make"

    vehicle_make_id: Mapped[int] = mapped_column(primary_key=True)
    make: Mapped[str] = mapped_column(String(31), unique=True)


class Directory(VehicleBase):
    __tablename__ = "directory"

    directory_id: Mapped[int] = mapped_column(primary_key=True)
    dirname: Mapped[str] = mapped_column(String(255), unique=True)

    videos: Mapped[list["Video"]] = relationship(back_populates="directory")
    images: Mapped[list["Image"]] = relationship(back_populates="directory")


class Video(VehicleBase):
    __tablename__ = "video"

    video_id: Mapped[int] = mapped_column(primary_key=True)
    directory_id: Mapped[int] = mapped_column(ForeignKey("directory.directory_id"))
    filename: Mapped[str] = mapped_column(String(63))
    observed_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))

    directory: Mapped[Directory] = relationship(back_populates="videos")
    images: Mapped[list["Image"]] = relationship(back_populates="video")
    vehicle: Mapped["Vehicle"] = relationship(back_populates="video")
    __table_args__ = (
        UniqueConstraint("directory_id", "filename", name="uix_directory_filename"),
    )


class Image(VehicleBase):
    __tablename__ = "image"

    image_id: Mapped[int] = mapped_column(primary_key=True)
    video_time: Mapped[Optional[float]]
    filename: Mapped[str] = mapped_column(String(63))
    observed_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))

    video_id: Mapped[Optional[int]] = mapped_column(ForeignKey("video.video_id"))
    directory_id: Mapped[int] = mapped_column(ForeignKey("directory.directory_id"))

    directory: Mapped[Directory] = relationship(back_populates="images")
    vehicle: Mapped["Vehicle"] = relationship(back_populates="vehicle_image")
    video: Mapped[Video] = relationship(back_populates="images")

    __table_args__ = (
        UniqueConstraint("directory_id", "filename", name="uix_directory_filename"),
    )


class Color(VehicleBase):
    __tablename__ = "color"

    color_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(15), unique=True)
    red: Mapped[int]
    green: Mapped[int]
    blue: Mapped[int]
    __table_args__ = (
        UniqueConstraint("red", "green", "blue", name="uix_color"),
    )


class Location(VehicleBase):
    __tablename__ = "location"

    location_id: Mapped[int] = mapped_column(primary_key=True)
    latitude: Mapped[float]
    longitude: Mapped[float]

    vehicle: Mapped["Vehicle"] = relationship(back_populates="location")
    __table_args__ = (
        UniqueConstraint("latitude", "longitude", name="uix_combined_location"),
    )


class VehicleType(VehicleBase):
    __tablename__ = "vehicle_type"

    vehicle_type_id: Mapped[int] = mapped_column(primary_key=True)
    vehicle_type: Mapped[str] = mapped_column(unique=True)


class Vehicle(VehicleBase):
    __tablename__ = "vehicle"

    vehicle_id: Mapped[int] = mapped_column(primary_key=True)

    # Direct data
    video_time: Mapped[float]
    is_taxed: Mapped[Optional[bool]]
    has_mot: Mapped[Optional[bool]]
    observed_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))

    # Relationships requiring join
    num_plate_id: Mapped[Optional[int]] = mapped_column(ForeignKey("number_plate.num_plate_id"))
    make_id: Mapped[Optional[int]] = mapped_column(ForeignKey("vehicle_make.vehicle_make_id"))
    color_id: Mapped[Optional[int]] = mapped_column(ForeignKey("color.color_id"))
    video_id: Mapped[int] = mapped_column(ForeignKey("video.video_id"))
    vehicle_image_id: Mapped[int] = mapped_column(ForeignKey("image.image_id"), unique=True)
    vehicle_type: Mapped[Optional[int]] = mapped_column(ForeignKey("vehicle_type.vehicle_type_id"))
    location_id: Mapped[int] = mapped_column(ForeignKey("location.location_id"))

    # Relationships
    video: Mapped[Video] = relationship(back_populates="vehicle")
    vehicle_image: Mapped[Image] = relationship(back_populates="vehicle")
    location: Mapped[Location] = relationship(back_populates="vehicle")

    # Unique constraints
    __table_args__ = (
        UniqueConstraint("video_id", "video_time", "vehicle_image_id", name="uix_video"),
    )
