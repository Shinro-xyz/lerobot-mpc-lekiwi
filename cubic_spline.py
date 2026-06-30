import numpy as np
from components import TrajectoryGenerator

class CubicSpline(TrajectoryGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, start_position: Any, end_position: Any, duration: Any) -> Any:
        return super().generate(start_position, end_position, duration)