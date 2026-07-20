import pytest
import numpy as np
from utils.array_backend import NumpyBackend


def _to_np(x, bk):
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestLQR:
    def test_lqr_gain_stabilizes_1d(self, bk):
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
        config = {"state_cost": [1.0, 1.0], "control_cost": [1.0, 1.0], "dt": 0.1}
        from controllers.lqr import LQR
        lqr = LQR.from_config(config, backend=bk)
        assert lqr.K is not None


class TestPID:
    def test_pid_derivative_zero_on_first_call(self, bk):
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
        config = {"kp": [1.0], "ki": [0.5], "kd": [0.1], "dt": 0.01}
        from controllers.pid import PIDController
        pid = PIDController.from_config(config, backend=bk)
        assert pid.kp is not None


class TestMPC:
    def test_mpc_H_symmetric(self, bk):
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
