import numpy as np
from components import TrajectoryGenerator

class CubicPolynomial(TrajectoryGenerator):
    def generate(self, start_position: np.ndarray, end_position: np.ndarray, duration: float):
        