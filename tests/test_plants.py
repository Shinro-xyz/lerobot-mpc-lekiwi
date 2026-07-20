import pytest
import numpy as np
from utils.array_backend import NumpyBackend


def _to_np(x, bk):
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestHolonomicMobileRobot:
    def test_A_is_identity(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        A, B = robot.get_model()
        assert np.allclose(_to_np(A, bk), np.eye(3))

    def test_B_is_dt_times_identity(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        dt = 0.05
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=dt, backend=bk)
        A, B = robot.get_model()
        assert np.allclose(_to_np(B, bk), dt * np.eye(3))

    def test_step_integrates_correctly(self, bk):
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
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        state = robot.get_state()
        state[0] = 99.0
        internal = robot.get_state()
        assert _to_np(internal, bk)[0] != 99.0

    def test_set_pose_updates_state(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        robot.set_pose(1.0, 2.0, 0.5)
        state = robot.get_state()
        assert np.allclose(_to_np(state, bk), [1.0, 2.0, 0.5])

    def test_step_returns_wheel_speeds(self, bk):
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        wheel_speeds = robot.step(u)
        assert _to_np(wheel_speeds, bk).shape == (3,)
