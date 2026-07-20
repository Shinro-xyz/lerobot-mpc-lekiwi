import pytest
from components import Controller, Plant, StateEstimator, TrajectoryGenerator, PhysicsEngine
from utils.array_backend import NumpyBackend


class TestABCs:
    def test_controller_abstract(self):
        with pytest.raises(TypeError):
            Controller()

    def test_plant_abstract(self):
        with pytest.raises(TypeError):
            Plant()

    def test_state_estimator_abstract(self):
        with pytest.raises(TypeError):
            StateEstimator()

    def test_trajectory_generator_abstract(self):
        with pytest.raises(TypeError):
            TrajectoryGenerator()

    def test_physics_engine_abstract(self):
        with pytest.raises(TypeError):
            PhysicsEngine()

    def test_physics_engine_default_backend(self):
        assert PhysicsEngine.backend.fget(None) is not None
        bk = PhysicsEngine.backend.fget(None)
        assert isinstance(bk, NumpyBackend)
