import os

is_mac = 'zsh' in os.getenv("SHELL")

sql_path: str = "sqlite:////home/matt/traffic_cop.db"
yolo_device: str | None = "cpu"

if is_mac:
    sql_path: str = "sqlite:///"
    yolo_device: str | None = "mps"

