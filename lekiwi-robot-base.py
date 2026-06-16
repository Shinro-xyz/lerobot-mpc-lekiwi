import numpy as np

class HolonomicMobileRobot:
    def __init__(self, num_wheels:int,radius_robots:float,gamma:float,radius_wheels:float ):
        """Covers holonomic robots, which lekiwi is """
        # n:number of wheels
        # gamma:first wheel angle from base of the robot, use the MJCF definition
        # R: is the robot’s radius / the distance between the robot’s center and the wheels.
        self.n=num_wheels
        self.R=radius_robots
        self.gamma=gamma
        self.r=radius_wheels

def MobileRobotBaseKinematics(n:int,R:float,gamma:float, r:float):

    """Covers holonomic robots, which lekiwi is """
    # n:number of wheels
    # gamma:first wheel angle from base of the robot, use the MJCF definition
    # R: is the robot’s radius / the distance between the robot’s center and the wheels.
    theta=2*np.pi/n
    angle_list=[]
    for i in range(n):
        angle=i*theta+gamma
        angle_list.append(angle)
    sin_list=np.sin(angle_list)
    cos_list=np.cos(angle_list)
    #inverse kinematics, LTI matrix
    A=np.column_stack((sin_list,-cos_list,np.full_like(sin_list, -R)))
    return A, np.linalg.pinv(A)
