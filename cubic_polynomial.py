import numpy as np
from components import TrajectoryGenerator

class CubicPolynomial(TrajectoryGenerator):
    def generate(self, start_position: np.ndarray, end_position: np.ndarray, duration: float, start_vel:np.ndarray, end_vel: np.ndarray):
        # define the trajectory coefficients
        self.a0= start_position
        self.a1= start_vel
        a2_numerator= 3*(end_position-start_position)-duration*(2*start_vel+end_vel)
        self.a2= a2_numerator/(duration**2)

        a3_numerator= -2*(end_position-end_position)+duration*(start_vel+end_vel)
        self.a3= a3_numerator/(duration**3)

        self.duration=duration

    def position_at(self, t: float):
        
