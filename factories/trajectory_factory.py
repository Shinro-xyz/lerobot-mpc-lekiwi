import tomllib
from factories.registry import _TRAJECTORY_REGISTRY


class TrajectoryFactory:
    """Creates a schedule array from TOML config files via the registry.

    Usage:
        schedule = TrajectoryFactory("configs/trajectories/base_straight.toml").create()
        setpoint = schedule[step]  # returns np.ndarray
    """

    def __init__(self, config_path: str):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

    def create(self):
        cls = _TRAJECTORY_REGISTRY[self.config["type"]]
        return cls.from_config(self.config)