import tomllib
from factories.registry import _CONTROLLER_REGISTRY


class ControllerFactory:
    """Creates Controller instances from TOML config files via the registry.

    Usage:
        ctrl = ControllerFactory("configs/controllers/lqr_base.toml").create()
    """

    def __init__(self, config_path: str):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

    def create(self):
        cls = _CONTROLLER_REGISTRY[self.config["type"]]
        return cls.from_config(self.config)