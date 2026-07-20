import pytest
import numpy as np
from utils.array_backend import NumpyBackend


def _to_np(x, bk):
    """Convert a backend array to numpy for assertion comparisons."""
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestHolonomicMobileRobot:
    """Verify holonomic mobile robot: state-space model, integration, copy semantics, and wheel speeds."""

    def test_A_is_identity(self, bk):
        """The discrete-time state matrix A is the 3x3 identity."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        A, B = robot.get_model()
        assert np.allclose(_to_np(A, bk), np.eye(3))

    def test_B_is_dt_times_identity(self, bk):
        """The discrete-time input matrix B is dt * I_3."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        dt = 0.05
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=dt, backend=bk)
        A, B = robot.get_model()
        assert np.allclose(_to_np(B, bk), dt * np.eye(3))

    def test_step_integrates_correctly(self, bk):
        """step() integrates the state: state += u * dt."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        dt = 0.01
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=dt, backend=bk)
        u = bk.array([0.5, 0.0, 0.0])
        robot.step(u)
        state = robot.get_state()
        expected = np.array([0.5 * dt, 0.0, 0.0])
        assert np.allclose(_to_np(state, bk), expected, atol=1e-10)

    def test_get_state_returns_copy(self, bk):
        """get_state() returns a copy, not a reference to the internal state."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        state = robot.get_state()
        state[0] = 99.0
        internal = robot.get_state()
        assert _to_np(internal, bk)[0] != 99.0

    def test_set_pose_updates_state(self, bk):
        """set_pose() directly sets the robot's pose."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        robot.set_pose(1.0, 2.0, 0.5)
        state = robot.get_state()
        assert np.allclose(_to_np(state, bk), [1.0, 2.0, 0.5])

    def test_step_returns_wheel_speeds(self, bk):
        """step() returns a wheel speed vector of length n_wheels."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        wheel_speeds = robot.step(u)
        assert _to_np(wheel_speeds, bk).shape == (3,)
