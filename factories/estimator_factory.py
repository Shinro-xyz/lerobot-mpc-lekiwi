import numpy as np
import yaml
from components import StateEstimator


class EstimatorFactory:
    """Creates StateEstimator instances from YAML config files.

    Usage:
        est = EstimatorFactory("estimators/luenberger_config.yaml").create()
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def _diag(self, key: str) -> np.ndarray:
        vals = self.config[key]
        return np.diag(vals) if isinstance(vals, list) else np.array(vals)

    def create(self) -> StateEstimator:
        etype = self.config["type"]
        n = len(self.config.get("state_covariance", []))

        if etype == "LuenbergerObserver":
            from estimators.luenberger_observer import LuenbergerObserver
            return LuenbergerObserver(
                A=np.eye(n),
                B=self.config.get("dt", 0.02) * np.eye(n),
                observer_gain=self._diag("observer_gain"),
                C=np.eye(n),
                D=np.zeros((n, n)),
                x0=np.zeros((n, 1)),
            )

        elif etype == "KalmanFilter":
            from estimators.kalman_filter import KalmanFilter
            dt = self.config.get("dt", 0.02)
            return KalmanFilter(
                A=np.eye(n),
                B=dt * np.eye(n),
                Q=self._diag("process_noise"),
                R=self._diag("measurement_noise"),
                C=np.eye(n),
                D=np.zeros((n, n)),
                x0=np.zeros((n, 1)),
            )

        else:
            raise ValueError(f"Unknown estimator type: {etype}")