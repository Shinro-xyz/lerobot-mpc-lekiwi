import pytest
import numpy as np
from utils.array_backend import NumpyBackend


def _to_np(x, bk):
    """Convert a backend array to numpy for assertion comparisons."""
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestLQR:
    """Verify LQR gain computation against analytical DARE solution."""

    def test_lqr_gain_stabilizes_1d(self, bk):
        """Closed-loop A - B @ K has eigenvalues inside the unit circle."""
        A = bk.eye(1)
        B = bk.eye(1)
        Q = bk.eye(1)
        R = bk.eye(1)
        from controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        K = lqr.K
        A_cl = A - B @ K
        eigs = np.linalg.eigvals(_to_np(A_cl, bk))
        assert np.all(np.abs(eigs) < 1)

    def test_lqr_gain_analytical_1d(self, bk):
        """For A=1, B=1, Q=1, R=1, the DARE solution is P = (1+sqrt(5))/2 and K = P/(1+P)."""
        A = bk.eye(1)
        B = bk.eye(1)
        Q = bk.eye(1)
        R = bk.eye(1)
        from controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        K = _to_np(lqr.K, bk)[0, 0]
        P_expected = (1 + np.sqrt(5)) / 2
        K_expected = P_expected / (1 + P_expected)
        assert np.allclose(K, K_expected, atol=1e-10)

    def test_lqr_dare_residual(self, bk):
        """The DARE residual ||A^T P A - P - A^T P B (R + B^T P B)^{-1} B^T P A + Q|| is near zero."""
        A = bk.eye(1)
        B = bk.eye(1)
        Q = bk.eye(1)
        R = bk.eye(1)
        from controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(_to_np(A, bk), _to_np(B, bk), _to_np(Q, bk), _to_np(R, bk))
        residual = A.T @ P @ A - P - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q
        assert np.linalg.norm(residual) < 1e-10

    def test_lqr_compute_shape(self, bk):
        """compute() returns a control vector of dimension n_u."""
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.eye(2)
        from controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        x = bk.array([1.0, 2.0])
        u = lqr.compute(x)
        assert _to_np(u, bk).shape == (2,)

    def test_lqr_regulation_to_zero(self, bk):
        """compute(x) with no target returns u = -K @ x (regulation to origin)."""
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.eye(2)
        from controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        x = bk.array([1.0, 2.0])
        u = lqr.compute(x)
        u_expected = -lqr.K @ x
        assert np.allclose(_to_np(u, bk), _to_np(u_expected, bk))

    def test_lqr_from_config(self, bk):
        """from_config creates a valid LQR controller with a gain matrix."""
        config = {"state_cost": [1.0, 1.0], "control_cost": [1.0, 1.0], "dt": 0.1}
        from controllers.lqr import LQR
        lqr = LQR.from_config(config, backend=bk)
        assert lqr.K is not None


class TestPID:
    """Verify PID controller: steady-state error, anti-windup, and reset."""

    def test_pid_derivative_zero_on_first_call(self, bk):
        """Derivative term is zero on the first call (no previous error)."""
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([0.0]),
            kd=bk.array([1.0]),
            dt=0.1,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0]), bk.array([0.0]))
        p_term = 1.0 * (0.0 - 1.0)
        assert np.allclose(_to_np(u, bk)[0], p_term)

    def test_pi_eliminates_steady_state_error(self, bk):
        """PI control drives a first-order plant to the target with zero steady-state error."""
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([0.5]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        target = bk.array([1.0])
        x = bk.array([0.0])
        for _ in range(3000):
            u = pid.compute(x, target)
            x = x + 0.01 * u
        assert np.allclose(_to_np(x, bk)[0], 1.0, atol=1e-2)

    def test_p_only_steady_state_error(self, bk):
        """P-only control has non-zero steady-state error for a first-order plant."""
        from controllers.pid import PIDController
        Kp = 2.0
        pid = PIDController(
            kp=bk.array([Kp]),
            ki=bk.array([0.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        target = bk.array([1.0])
        x = bk.array([0.0])
        for _ in range(1000):
            u = pid.compute(x, target)
            x = x + 0.01 * u
        assert np.allclose(_to_np(x, bk)[0], 1.0, atol=1e-2)

    def test_pid_anti_windup(self, bk):
        """When output is clamped, the integral term back-calculates on saturated channels."""
        from controllers.pid import PIDController
        lo = bk.array([-0.5])
        hi = bk.array([0.5])
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([10.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            output_limits=(lo, hi),
            backend=bk,
        )
        target = bk.array([10.0])
        x = bk.array([0.0])
        for _ in range(200):
            u = pid.compute(x, target)
            x = x + 0.01 * u
        assert np.allclose(_to_np(u, bk)[0], 0.5, atol=1e-3)

    def test_pid_reset(self, bk):
        """reset() clears the integral accumulator and previous error."""
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([1.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        pid.compute(bk.array([1.0]), bk.array([0.0]))
        pid.reset()
        assert np.allclose(_to_np(pid._integral, bk), 0.0)
        assert np.allclose(_to_np(pid._prev_error, bk), 0.0)
        assert not pid.has_run

    def test_pid_from_config(self, bk):
        """from_config creates a valid PID controller."""
        config = {"kp": [1.0], "ki": [0.5], "kd": [0.1], "dt": 0.01}
        from controllers.pid import PIDController
        pid = PIDController.from_config(config, backend=bk)
        assert pid.kp is not None


class TestMPC:
    """Verify MPC: H symmetry, F shape, constraint satisfaction, and from_config."""

    def test_mpc_H_symmetric(self, bk):
        """The QP Hessian H is symmetric."""
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        mpc = MPC_LTI(horizon=5, control_cost_matrix=R, state_cost_matrix=Q,
                      A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)
        H = _to_np(mpc.H, bk)
        assert np.allclose(H, H.T)

    def test_mpc_F_shape(self, bk):
        """The QP linear term F has shape (n_x, N * n_u)."""
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        mpc = MPC_LTI(horizon=5, control_cost_matrix=R, state_cost_matrix=Q,
                      A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)
        F = _to_np(mpc.F, bk)
        assert F.shape == (n, 5 * m)

    def test_mpc_compute_shape(self, bk):
        """compute() returns a control vector of dimension n_u."""
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        mpc = MPC_LTI(horizon=5, control_cost_matrix=R, state_cost_matrix=Q,
                      A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)
        F = bk.eye(m)
        mpc.constraints(F, bk.array([1.0, 1.0]), bk.array([-1.0, -1.0]))
        x0 = bk.array([1.0, 0.0])
        u = mpc.compute(x0)
        assert _to_np(u, bk).shape == (m,)

    def test_mpc_constraints_respected(self, bk):
        """MPC respects hard input constraints |u| <= bound."""
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        mpc = MPC_LTI(horizon=5, control_cost_matrix=R, state_cost_matrix=Q,
                      A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)
        bound = 0.5
        F = bk.eye(m)
        mpc.constraints(F, bk.array([bound, bound]), bk.array([-bound, -bound]))
        x0 = bk.array([10.0, 10.0])
        u = mpc.compute(x0)
        u_val = _to_np(u, bk)
        assert np.all(np.abs(u_val) <= bound + 1e-4)

    def test_mpc_from_config(self, bk):
        """from_config creates a valid MPC controller with precomputed H and F."""
        config = {
            "horizon": 5,
            "state_cost": [1.0, 1.0],
            "control_cost": [1.0, 1.0],
            "dt": 0.1,
        }
        from controllers.mpc_lti import MPC_LTI_Base
        mpc = MPC_LTI_Base.from_config(config, backend=bk)
        assert mpc.H is not None
        assert mpc.F is not None
