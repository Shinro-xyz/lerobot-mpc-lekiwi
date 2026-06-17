import numpy as np
from components import Controller
from scipy.linalg import solve_discrete_are

class LQR(Controller):
    """
    Linear Quadratic Regulator (LQR) controller for discrete-time systems.
    """
    def __init__(self, state_cost_matrix:np.ndarray, control_cost_matrix: np.ndarray,dynamics_state_matrix: np.ndarray, dynamics_control_matrix: np.ndarray):
        """
        Initialize the LQR controller.

        Args:
            state_cost_matrix: Q matrix penalizing state deviation.
            control_cost_matrix: R matrix penalizing control effort.
            dynamics_state_matrix: A matrix representing system dynamics.
            dynamics_control_matrix: B matrix representing control input influence.
        """
        self.A=dynamics_state_matrix
        self.B=dynamics_control_matrix
        self.Q=state_cost_matrix
        self.R=control_cost_matrix
        self.gain_calculation()

    def gain_calculation(self):
        """
        Calculate the optimal LQR gain matrix K using the discrete-time algebraic Riccati equation.
        """
        P=solve_discrete_are(self.A,self.B,self.Q,self.R)
        self.K=np.linalg.inv(self.R+self.B.T@P@self.B)@(self.B.T@P@self.A)

    def compute(self, current_state:np.ndarray, target_state=None):
        """
        Compute the control input for the given current state.

        Args:
            current_state: The current state of the system.
            target_state: The desired target state. Defaults to zeros.

        Returns:
            The optimal control input.
        """
        if target_state is None:

            target_state=np.zeros_like(current_state)
            error=target_state-current_state
        else:
            error=target_state-current_state

        return self.K@error
    def reset(self):
        """
        Reset the controller state.
        """
        pass

        
    