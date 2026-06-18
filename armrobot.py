import numpy as np
from components import Plant

class ArmRobot(Plant):
    def __init__(self, num_dof:int, dt:float, joint_limits:np.ndarray,joint_offsets:np.ndarray):
        self.num_dof=num_dof
        self.dt=dt
        self.state=np.zeros(num_dof)
        self.joint_offsets=joint_offsets
        self.joint_limits=joint_limits

    def get_state(self):
        return self.state.copy()

    def get_model(self):
        A=np.eye(self.num_dof)
        B=self.dt*np.eye(self.num_dof)
        return A,B
    def _homogenous_transform(self,joint_angles:np.ndarray,axes:list[str]):
        sines,cosines= np.sin(joint_angles), np.cos(joint_angles)
        T=np.zeros((self.num_dof,4,4))
        for i in range(self.num_dof):
            axis=axes[i]
            s,c=sines[i], cosines[i]
            if axis=='x':
                R=np.array([[1,0,0],[0,c,-s],[0,s,c]])
            elif axis=='y':
                R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            elif axis=='z':
                R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            else:
                raise ValueError(f"Invalid axis '{axis}' at joint {i}. Choose 'x', 'y', or 'z'.")
            offset_vector= self.joint_offsets[i,:3].reshape(3,1)
            T[:3,:3]=R
            T[:3,3]=offset_vector
            T[3,3]=1.0
        return T
    def forward_kinematics(self,joint_angles:np.ndarray):
        