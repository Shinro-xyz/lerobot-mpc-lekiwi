import numpy as np
from components import TrajectoryGenerator

class QuintiPolynomial(TrajectoryGenerator):
    def generate(self, 
        start_position: np.ndarray, 
        end_position:np.ndarray, 
        duration:float,
        start_vel: np.ndarray= None,
        end_vel: np.ndarray= None,
        start_acc: np.ndarray= None,
        end_acc: np.ndarray= None
    ):
        
        
