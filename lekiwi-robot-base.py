import numpy as np
from numpy._core.strings import lower
from scipy.sparse import block_diag
import osqp
from scipy import sparse

class HolonomicMobileRobot:
    def __init__(self, num_wheels:int,radius_robots:float,gamma:float,radius_wheels:float, dt:float):
        """Covers holonomic robots, which lekiwi is """
        # n:number of wheels
        # gamma:first wheel angle from base of the robot, use the MJCF definition
        # R: is the robot’s radius / the distance between the robot’s center and the wheels.
        self.n=num_wheels
        self.R=radius_robots
        self.gamma=gamma
        self.r=radius_wheels
        self.dt=dt
        self.state=np.zeros(3)

    def mobilerobotkinematics(self):
        theta_perwheel=2*np.pi/self.n
        angle_list=[]
        for i in range(self.n):
            angle=i*theta_perwheel+self.gamma
            angle_list.append(angle)
        sin_list=np.sin(angle_list)
        cos_list=np.cos(angle_list)
        #inverse kinematics, LTI matrix
        A_kin=np.column_stack((sin_list,-cos_list,np.full_like(sin_list, -self.R)))
        return A_kin, np.linalg.pinv(A_kin)

    def step(self,u_world):
        theta=self.state[2]
        c, s= np.cos(theta), np.sin(theta)
        rot_matrix= np.array([[c,s,0],[-s,c,0],[0,0,1]])
        A_kinematics, A_pinv_kin=self.mobilerobotkinematics()
        u_body=rot_matrix@u_world
        wheel_speeds=(1/self.r)*A_kinematics*u_body
        self.state+=u_world*self.dt
        return wheel_speeds

    def set_pose(self,x,y,theta):
        self.state=np.array([x,y,theta])

class MPC_LTI:
    def __init__(self,horizon:int, control_cost_matrix:np.ndarray, state_cost_matrix:np.ndarray,A_dynamics:np.ndarray,B_dynamics: np.ndarray, terminal_cost:np.ndarray):
        self.N=horizon
        self.Q=state_cost_matrix
        self.R=control_cost_matrix
        self.A=A_dynamics
        self.B=B_dynamics
        self.P=terminal_cost

    def constraits(self, constraint_matrix: np.ndarray,upper_bounds:np.ndarray, lower_bounds:np.ndarray):
        self.A_constraints=constraint_matrix
        self.lcons=lower_bounds
        self.ucons=upper_bounds
    
    def _mpc_dynamics_matrices(self):
        self.n=self.A.shape[0] #state dim
        m= self.B.shape[1] #control dim

        T_list=[]
        for n_step in range(self.N):
            A_new=np.linalg.matrix_power(self.A, n_step)
            T_list.append(A_new)
        self.T_bar=np.vstack(T_list)

        self.S_bar=np.zeros((self.N*self.n,self.N*m))
        for i in range(self.N):
            for j in range(i+1):
                self.S_bar[i*self.n:(i+1)*self.n, j*m:(j+1)*m] = np.linalg.matrix_power(self.A, i-j) @ self.B

    def _mpc_cost_matrices(self):
        # Q_bar: block diagonal of Q (with P at the end for terminal cost)
        Q_bar = np.zeros((self.N * self.n, self.N * self.n))
        for i in range(self.N - 1):
            Q_bar[i*self.n:(i+1)*self.n, i*self.n:(i+1)*self.n] = self.Q
        Q_bar[(self.N-1)*self.n:self.N*self.n, (self.N-1)*self.n:self.N*self.n] = self.P  # terminal cost
        
        # R_bar: block diagonal of R
        R_bar = np.kron(np.eye(self.N), self.R)  
        self.H=2*(R_bar+self.S_bar.T@Q_bar@self.S_bar)
        self.F=2*(self.T_bar@Q_bar@self.S_bar)

    def solve(self,x0):
        prob=osqp.OSQP()
        q=self.F.T@x0
        prob.setup(sparse.csc_matrix(self.H), q.flatten(),self.A_constraints, self.lcons,self.u)
        

    


        
        
        