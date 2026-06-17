import numpy as np
from components import Plant

class ArmRobot(Plant):
    def __init__(self, num_dof:int, dt:float):
        self.num_dof=num_dof
        self.dt=dt
        self.state=np.zeros(num_dof)

    def get_state(self):
        return self.state.copy()

    def get_model
        