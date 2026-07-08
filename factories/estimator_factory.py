import tomllib
from factories.registry import _ESTIMATOR_REGISTRY


class EstimatorFactory:
    """Creates StateEstimator instances from TOML config files via the registry.

    Usage:
        est = EstimatorFactory("configs/estimators/luenberger_base.toml").create()
    """

    def __init__(self, config_path: str):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

    def create(self):
        cls = _ESTIMATOR_REGISTRY[self.config["type"]]
        return cls.from_config(self.config)