import pytest
import numpy as np
from scipy.sparse import csr_matrix
from utils.array_backend import NumpyBackend


def _to_np(x, bk):
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


# =============================================================================
# Array Backend
# =============================================================================

class TestArrayBackendAdversarial:
    def setup_method(self):
        self.bk = NumpyBackend()

    def test_inv_nan(self):
        A = np.array([[np.nan, 0], [0, 1]], dtype=float)
        result = self.bk.inv(A)
        assert np.any(np.isnan(result))

    def test_inv_inf(self):
        A = np.array([[np.inf, 0], [0, 1]], dtype=float)
        result = self.bk.inv(A)
        assert np.all(np.isfinite(result))

    def test_inv_singular(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.inv(A)

    def test_inv_non_square(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.inv(A)

    def test_cholesky_non_psd(self):
        A = np.array([[-1, 0], [0, -1]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.cholesky(A)

    def test_cholesky_non_square(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.cholesky(A)

    def test_solve_singular(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        b = np.array([1, 2], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.solve(A, b)

    def test_solve_non_square(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        b = np.array([1, 2], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.solve(A, b)

    def test_svd_nan(self):
        A = np.array([[np.nan, 0], [0, 1]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.svd(A)

    def test_eigvals_nan(self):
        A = np.array([[np.nan, 0], [0, 1]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.eigvals(A)

    def test_matrix_rank_zero(self):
        A = np.zeros((5, 5))
        assert self.bk.matrix_rank(A) == 0

    def test_matrix_rank_empty(self):
        A = np.zeros((0, 0))
        assert self.bk.matrix_rank(A) == 0

    def test_cond_singular(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        assert self.bk.cond(A) > 1e15

    def test_cond_zero(self):
        A = np.zeros((3, 3))
        assert self.bk.cond(A) == np.inf

    def test_sqrt_negative(self):
        x = np.array([-1.0])
        result = self.bk.sqrt(x)
        assert np.isnan(result[0])

    def test_arccos_out_of_range(self):
        result = self.bk.arccos(np.array([2.0]))
        assert np.any(np.isnan(result))

    def test_zeros_empty_shape(self):
        z = self.bk.zeros(0)
        assert z.shape == (0,)

    def test_zeros_negative_shape(self):
        with pytest.raises(ValueError):
            self.bk.zeros(-1)

    def test_eye_zero(self):
        I = self.bk.eye(0)
        assert I.shape == (0, 0)

    def test_eye_negative(self):
        with pytest.raises(ValueError):
            self.bk.eye(-1)

    def test_linspace_reversed(self):
        x = self.bk.linspace(1, 0, 5)
        assert _to_np(x, self.bk)[0] == 1.0
        assert _to_np(x, self.bk)[-1] == 0.0

    def test_linspace_zero_points(self):
        x = self.bk.linspace(0, 1, 0)
        assert len(x) == 0

    def test_linspace_single_point(self):
        x = self.bk.linspace(0, 1, 1)
        assert len(x) == 1

    def test_matrix_power_zero(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        A0 = self.bk.matrix_power(A, 0)
        assert np.allclose(A0, np.eye(2))

    def test_matrix_power_negative(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.bk.matrix_power(A, -1)
        assert np.allclose(result @ A, np.eye(2))

    def test_matrix_power_non_square(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError):
            self.bk.matrix_power(A, 2)

    def test_vstack_empty_list(self):
        with pytest.raises(ValueError):
            self.bk.vstack([])

    def test_hstack_empty_list(self):
        with pytest.raises(ValueError):
            self.bk.hstack([])

    def test_vstack_mismatched_cols(self):
        a = np.array([1, 2], dtype=float)
        b = np.array([3, 4, 5], dtype=float)
        with pytest.raises(ValueError):
            self.bk.vstack([a, b])

    def test_hstack_mismatched_rows(self):
        a = np.array([[1], [2]], dtype=float)
        b = np.array([[3]], dtype=float)
        with pytest.raises(ValueError):
            self.bk.hstack([a, b])

    def test_reshape_incompatible(self):
        x = np.array([1, 2, 3], dtype=float)
        with pytest.raises(ValueError):
            self.bk.reshape(x, 2, 2)

    def test_tile_negative_reps(self):
        x = np.array([1, 2], dtype=float)
        with pytest.raises(ValueError):
            self.bk.tile(x, -1)

    def test_clip_swapped_bounds(self):
        x = np.array([0.5], dtype=float)
        clipped = self.bk.clip(x, 1.0, 0.0)
        assert _to_np(clipped, self.bk)[0] == 0.0

    def test_where_mismatched_shapes(self):
        cond = np.array([True, False])
        a = np.array([1, 2, 3], dtype=float)
        b = np.array([10, 20], dtype=float)
        with pytest.raises(ValueError):
            self.bk.where(cond, a, b)

    def test_block_mismatched_shapes(self):
        A = np.array([[1, 2]], dtype=float)
        B = np.array([[3]], dtype=float)
        C = np.array([[4]], dtype=float)
        with pytest.raises(ValueError):
            self.bk.block([[A, B], [C]])

    def test_kron_empty(self):
        a = np.array([], dtype=float)
        b = np.array([1, 2], dtype=float)
        k = self.bk.kron(a, b)
        assert len(k) == 0

    def test_cross_2d(self):
        a = np.array([1, 0], dtype=float)
        b = np.array([0, 1], dtype=float)
        result = self.bk.cross(a, b)
        assert np.isscalar(result) or result.shape == ()

    def test_cross_4d(self):
        a = np.array([1, 0, 0, 0], dtype=float)
        b = np.array([0, 1, 0, 0], dtype=float)
        with pytest.raises(ValueError):
            self.bk.cross(a, b)

    def test_norm_zero_vector(self):
        v = np.array([0, 0, 0], dtype=float)
        assert self.bk.norm(v) == 0.0

    def test_norm_empty(self):
        v = np.array([], dtype=float)
        assert self.bk.norm(v) == 0.0

    def test_diag_scalar(self):
        with pytest.raises(ValueError):
            self.bk.diag(5.0)

    def test_any_empty(self):
        assert not self.bk.any(np.array([], dtype=bool))

    def test_sum_empty(self):
        assert self.bk.sum(np.array([], dtype=float)) == 0.0

    def test_real_complex(self):
        x = np.array([1 + 2j, 3 + 4j])
        r = self.bk.real(x)
        assert np.allclose(r, [1, 3])

    def test_sort_reverse(self):
        x = np.array([3, 1, 2], dtype=float)
        assert np.allclose(self.bk.sort(x), [1, 2, 3])

    def test_abs_negative(self):
        x = np.array([-1, -2, -3], dtype=float)
        assert np.allclose(self.bk.abs(x), [1, 2, 3])

    def test_ravel_0d(self):
        x = np.array(5.0)
        y = self.bk.ravel(x)
        assert y.shape == (1,)

    def test_reshape_0d(self):
        x = np.array(5.0)
        y = self.bk.reshape(x, 1)
        assert y.shape == (1,)

    def test_copy_modification_independence(self):
        x = np.array([1, 2, 3], dtype=float)
        y = self.bk.copy(x)
        y[0] = 99
        assert x[0] == 1
        x[1] = 88
        assert y[1] == 2


# =============================================================================
# Controllability Checker
# =============================================================================

class TestControllabilityCheckerAdversarial:
    def test_zero_system(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        n = 3
        A = np.zeros((n, n))
        B = np.zeros((n, 1))
        C = np.eye(n)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert not ana.is_controllable()
        assert ana.is_observable()

    def test_identity_system(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        n = 3
        A = np.eye(n)
        B = np.eye(n)
        C = np.eye(n)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert ana.is_controllable()
        assert ana.is_observable()

    def test_scalar_system(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[-1.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert ana.is_controllable()
        assert ana.is_observable()
        Wc = ana.controllability_gramian()
        assert np.allclose(Wc, 0.5)

    def test_jordan_block(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        B = np.array([[0], [0], [1]], dtype=float)
        C = np.eye(3)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert ana.is_controllable()
        assert ana.is_observable()

    def test_jordan_block_uncontrollable(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        B = np.array([[1], [0], [0]], dtype=float)
        C = np.eye(3)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert not ana.is_controllable()

    def test_discrete_dt_zero(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[0.9, 0.1], [0, 0.8]], dtype=float)
        B = np.array([[0], [0.1]], dtype=float)
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, dt=0.0, backend=NumpyBackend())
        Wc = ana.discrete_controllability_gramian()
        assert np.all(np.linalg.eigvals(Wc) > -1e-10)

    def test_sparse_A(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        A = csr_matrix(np.array([[0, 1], [-1, -2]], dtype=float))
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert ana.is_controllable()

    def test_gramian_condition_singular(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.zeros((2, 2))
        B = np.array([[1], [1]], dtype=float)
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        with pytest.raises(ValueError):
            ana.gramian_condition("Wc")

    def test_gramian_spectrum_finite_horizon_malformed(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        with pytest.raises(ValueError):
            ana.gramian_spectrum("Wc_finite_bad")

    def test_balanced_truncate_r_equals_n(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        Ar, Br, Cr, Dr = ana.balanced_truncate(2)
        assert Ar.shape == (2, 2)

    def test_rank_report_on_zero_system(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.zeros((3, 3))
        B = np.zeros((3, 1))
        C = np.zeros((2, 3))
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        report = ana.rank_report()
        assert report["controllability"][0] == 0
        assert report["observability"][0] == 0

    def test_summary_on_zero_system(self):
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.zeros((2, 2))
        B = np.zeros((2, 1))
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        with pytest.raises(ValueError):
            ana.summary()


# =============================================================================
# Controllers
# =============================================================================

class TestLQRAdversarial:
    def test_lqr_zero_Q(self, bk):
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.zeros((2, 2))
        R = bk.eye(2)
        with pytest.raises(np.linalg.LinAlgError):
            LQR(Q, R, A, B, backend=bk)

    def test_lqr_zero_R(self, bk):
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.zeros((2, 2))
        lqr = LQR(Q, R, A, B, backend=bk)
        K = _to_np(lqr.K, bk)
        assert np.all(np.isfinite(K))

    def test_lqr_non_psd_Q(self, bk):
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.array([[-1, 0], [0, -1]])
        R = bk.eye(2)
        with pytest.raises(Exception):
            LQR(Q, R, A, B, backend=bk)

    def test_lqr_mismatched_dims(self, bk):
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(3)
        Q = bk.eye(2)
        R = bk.eye(2)
        with pytest.raises(Exception):
            LQR(Q, R, A, B, backend=bk)

    def test_lqr_compute_with_target(self, bk):
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.eye(2)
        lqr = LQR(Q, R, A, B, backend=bk)
        x = bk.array([1.0, 2.0])
        target = bk.array([3.0, 4.0])
        u = lqr.compute(x, target)
        u_expected = -lqr.K @ (x - target)
        assert np.allclose(_to_np(u, bk), _to_np(u_expected, bk))

    def test_lqr_compute_wrong_shape(self, bk):
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.eye(2)
        lqr = LQR(Q, R, A, B, backend=bk)
        x = bk.array([1.0, 2.0, 3.0])
        with pytest.raises(Exception):
            lqr.compute(x)


class TestPIDAdversarial:
    def test_pid_zero_gains(self, bk):
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([0.0]),
            ki=bk.array([0.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0]), bk.array([0.0]))
        assert np.allclose(_to_np(u, bk), 0.0)

    def test_pid_negative_kp(self, bk):
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([-1.0]),
            ki=bk.array([0.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        x = bk.array([0.0])
        target = bk.array([1.0])
        for _ in range(100):
            u = pid.compute(x, target)
            x = x + 0.01 * u
        assert _to_np(x, bk)[0] < 0

    def test_pid_mismatched_gain_lengths(self, bk):
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0, 2.0]),
            ki=bk.array([0.5]),
            kd=bk.array([0.1]),
            dt=0.01,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0, 2.0]), bk.array([0.0, 0.0]))
        assert _to_np(u, bk).shape == (2,)

    def test_pid_zero_dt(self, bk):
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([0.5]),
            kd=bk.array([0.0]),
            dt=0.0,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0]), bk.array([0.0]))
        assert not np.any(np.isnan(_to_np(u, bk)))

    def test_pid_negative_dt(self, bk):
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([0.5]),
            kd=bk.array([0.0]),
            dt=-0.01,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0]), bk.array([0.0]))
        assert not np.any(np.isnan(_to_np(u, bk)))

    def test_pid_wrong_state_shape(self, bk):
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([0.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0, 2.0]), bk.array([0.0]))
        assert _to_np(u, bk).shape == (2,)


class TestMPCAdversarial:
    def test_mpc_horizon_1(self, bk):
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        mpc = MPC_LTI(horizon=1, control_cost_matrix=R, state_cost_matrix=Q,
                      A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)
        F = bk.eye(m)
        mpc.constraints(F, bk.array([1.0, 1.0]), bk.array([-1.0, -1.0]))
        x0 = bk.array([1.0, 0.0])
        u = mpc.compute(x0)
        assert _to_np(u, bk).shape == (m,)

    def test_mpc_mismatched_B_dims(self, bk):
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 3
        A = bk.eye(n)
        B = bk.eye(m)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        with pytest.raises(Exception):
            MPC_LTI(horizon=5, control_cost_matrix=R, state_cost_matrix=Q,
                    A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)

    def test_mpc_no_constraints_crashes(self, bk):
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
        x0 = bk.array([1.0, 0.0])
        with pytest.raises(AttributeError):
            mpc.compute(x0)

    def test_mpc_zero_horizon(self, bk):
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        with pytest.raises(Exception):
            MPC_LTI(horizon=0, control_cost_matrix=R, state_cost_matrix=Q,
                    A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)

    def test_mpc_negative_horizon(self, bk):
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        with pytest.raises(Exception):
            MPC_LTI(horizon=-1, control_cost_matrix=R, state_cost_matrix=Q,
                    A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)


# =============================================================================
# Estimators
# =============================================================================

class TestKalmanFilterAdversarial:
    def test_kalman_zero_noise(self, bk):
        from estimators.kalman_filter import KalmanFilter
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = bk.zeros((n, n))
        R = bk.zeros((n, n))
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        x = kf.estimate(y, u)
        assert not np.any(np.isnan(_to_np(x, bk)))

    def test_kalman_infinite_measurement_noise(self, bk):
        from estimators.kalman_filter import KalmanFilter
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = 0.1 * bk.eye(n)
        R = 1e10 * bk.eye(n)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        x = kf.estimate(y, u)
        assert not np.any(np.isnan(_to_np(x, bk)))

    def test_kalman_infinite_process_noise(self, bk):
        from estimators.kalman_filter import KalmanFilter
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = 1e10 * bk.eye(n)
        R = 0.1 * bk.eye(n)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        x = kf.estimate(y, u)
        assert not np.any(np.isnan(_to_np(x, bk)))

    def test_kalman_mismatched_dims(self, bk):
        from estimators.kalman_filter import KalmanFilter
        n = 2
        A = bk.eye(n)
        B = bk.eye(n)
        Q = 0.1 * bk.eye(n)
        R = 0.1 * bk.eye(3)
        C = bk.eye(3)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        with pytest.raises(ValueError):
            kf.estimate(y, u)

    def test_kalman_non_psd_Q(self, bk):
        from estimators.kalman_filter import KalmanFilter
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = bk.array([[-1, 0], [0, -1]])
        R = 0.1 * bk.eye(n)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        x = kf.estimate(y, u)
        assert not np.any(np.isnan(_to_np(x, bk)))


class TestLuenbergerObserverAdversarial:
    def test_luenberger_zero_gain(self, bk):
        from estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        L = bk.zeros((n, n))
        C = bk.eye(n)
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        x = obs.estimate(y, u)
        assert not np.any(np.isnan(_to_np(x, bk)))

    def test_luenberger_destabilizing_gain(self, bk):
        from estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        L = 2.0 * bk.eye(n)
        C = bk.eye(n)
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        A_cl = A - L @ C
        eigs = np.linalg.eigvals(_to_np(A_cl, bk))
        assert np.any(np.abs(eigs) >= 1)

    def test_luenberger_mismatched_gain(self, bk):
        from estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        L = bk.eye(3)
        C = bk.eye(n)
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        with pytest.raises(ValueError):
            obs.estimate(y, u)


# =============================================================================
# Trajectories
# =============================================================================

class TestCubicPolynomialAdversarial:
    def test_cubic_zero_duration(self, bk):
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.0])
        vf = bk.array([0.0])
        traj.generate(p0, pf, 0.0, v0, vf)
        pos, _, _ = traj.position_at(0.0)
        assert np.any(np.isnan(_to_np(pos, bk)))

    def test_cubic_negative_duration(self, bk):
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.0])
        vf = bk.array([0.0])
        traj.generate(p0, pf, -1.0, v0, vf)
        pos, _, _ = traj.position_at(0.0)
        assert np.isfinite(_to_np(pos, bk)[0])

    def test_cubic_mismatched_dims(self, bk):
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0, 0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.0, 0.0])
        vf = bk.array([0.0])
        traj.generate(p0, pf, 1.0, v0, vf)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (2,)

    def test_cubic_zero_dimensional(self, bk):
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([])
        pf = bk.array([])
        v0 = bk.array([])
        vf = bk.array([])
        traj.generate(p0, pf, 1.0, v0, vf)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (0,)

    def test_cubic_large_time(self, bk):
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.0])
        vf = bk.array([0.0])
        traj.generate(p0, pf, 1.0, v0, vf)
        pos, _, _ = traj.position_at(1e6)
        assert np.allclose(_to_np(pos, bk), 1.0)


class TestQuinticPolynomialAdversarial:
    def test_quintic_zero_duration(self, bk):
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        with pytest.raises(Exception):
            traj.generate(p0, pf, 0.0)

    def test_quintic_negative_duration(self, bk):
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        traj.generate(p0, pf, -1.0)
        pos, _, _ = traj.position_at(0.0)
        assert np.isfinite(_to_np(pos, bk)[0])

    def test_quintic_mismatched_dims(self, bk):
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0, 0.0])
        pf = bk.array([1.0])
        with pytest.raises(Exception):
            traj.generate(p0, pf, 1.0)

    def test_quintic_zero_dimensional(self, bk):
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([])
        pf = bk.array([])
        traj.generate(p0, pf, 1.0)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (0,)

    def test_quintic_large_time(self, bk):
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        traj.generate(p0, pf, 1.0)
        pos, _, _ = traj.position_at(1e6)
        assert np.allclose(_to_np(pos, bk), 1.0)


# =============================================================================
# Plants
# =============================================================================

class TestHolonomicMobileRobotAdversarial:
    def test_zero_dt(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.0, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        robot.step(u)
        state = robot.get_state()
        assert np.allclose(_to_np(state, bk), 0.0)

    def test_negative_dt(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=-0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        robot.step(u)
        state = robot.get_state()
        assert _to_np(state, bk)[0] < 0

    def test_zero_wheel_radius(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.0, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        with pytest.raises(ZeroDivisionError):
            robot.step(u)

    def test_zero_robot_radius(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.0, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        wheel_speeds = robot.step(u)
        assert _to_np(wheel_speeds, bk).shape == (3,)

    def test_single_wheel(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=1, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        wheel_speeds = robot.step(u)
        assert _to_np(wheel_speeds, bk).shape == (1,)

    def test_many_wheels(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=10, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        wheel_speeds = robot.step(u)
        assert _to_np(wheel_speeds, bk).shape == (10,)

    def test_wrong_input_shape(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0])
        with pytest.raises(Exception):
            robot.step(u)

    def test_set_pose_then_step(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        robot.set_pose(1.0, 2.0, 0.5)
        u = bk.array([0.0, 0.0, 0.0])
        robot.step(u)
        state = robot.get_state()
        assert np.allclose(_to_np(state, bk), [1.0, 2.0, 0.5])
