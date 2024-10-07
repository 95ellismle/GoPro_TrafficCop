import os

sql_path: str = "sqlite:////home/matt/traffic_cop.db"
yolo_device: str | None = "mps" if 'zsh' in os.getenv('SHELL') else "cpu"

