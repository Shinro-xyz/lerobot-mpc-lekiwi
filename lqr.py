import numpy as np
from components import Controller
from scipy.linalg import solve_discrete_are

class LQR(Controller):
    def __init__(self) -> None:
        super().__init__()