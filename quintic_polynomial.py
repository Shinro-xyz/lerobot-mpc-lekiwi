import numpy as np
from components import TrajectoryGenerator

class QuinticPolynomial(TrajectoryGenerator):
    def generate(self, 
        start_position: np.ndarray, 
        end_position:np.ndarray, 
        duration:float,
        start_vel: np.ndarray= None,
        end_vel: np.ndarray= None,
        start_acc: np.ndarray= None,
        end_acc: np.ndarray= None
    ):
        start_vel=np.zeros_like(start_position) if start_vel is None else start_vel
        start_acc=np.zeros_like(start_position) if start_acc is None else start_acc
        end_vel=np.zeros_like(end_position) if end_vel is None else end_vel
        end_acc=np.zeros_like(end_position) if end_acc is None else end_acc

        T=duration
        M=np.array([
            [0,0,0,0,0,1],
            [T**5,T**4,T**3,T**2,T,1],
            [0,0,0,0,1,0],
            [5*T**4, 4*T**3, 3*T**2,2*T,1,0],
            [0,0,0,2,0,0],
            [20*T**3, 12*T**2,6*T,2,0,0]
        ])

        b=[start_position,end_position,start_vel, end_vel,start_acc,end_acc]

        ## finding the coeffieicents
        coeff_vectors=np.linalg.solve(M,b)

        self.A=coeff_vectors[0]
        self.B=coeff_vectors[1]
        
        
        
        
        
