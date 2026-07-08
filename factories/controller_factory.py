import numpy as np
import yaml
from pathlib import Path
from components import Controller


class ControllerFactory:
    """Creates Controller instances from YAML config files.

    Usage:
        ctrl = ControllerFactory("controllers/lqr_config.yaml").create()
        ctrl = ControllerFactory("controllers/mpc_config.yaml").create()
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def _diag(self, key: str) -> np.ndarray:
        vals = self.config[key]
        return np.diag(vals)

    def create(self) -> Controller:
        ctype = self.config["type"]
        n = len(self.config.get("state_cost", []))

        if ctype == "LQR":
            from controllers.lqr import LQR
            return LQR(
                state_cost_matrix=self._diag("state_cost"),
                control_cost_matrix=self._diag("control_cost"),
                dynamics_state_matrix=np.eye(n),
                dynamics_control_matrix=self.config.get("dt", 0.02) * np.eye(n),
            )

        elif ctype == "MPC_DeltaU":
            from controllers.mpc_lti import MPC_LTI_DeltaU
            dt = self.config.get("dt", 0.02)
            ctrl = MPC_LTI_DeltaU(
                delta_u_penalty=self._diag("delta_u_penalty"),
                horizon=self.config["horizon"],
                control_cost_matrix=self._diag("control_cost"),
                state_cost_matrix=self._diag("state_cost"),
                A_dynamics=np.eye(n),
                B_dynamics=dt * np.eye(n),
                terminal_cost=self._diag("state_cost"),
            )
            constraints = self.config.get("constraints")
            if constraints:
                F = np.vstack([np.eye(n), -np.eye(n)])
                ctrl.constraints(
                    F,
                    np.array(constraints["upper"]),
                    np.array(constraints["lower"]),
                )
            return ctrl

        else:
            raise ValueError(f"Unknown controller type: {ctype}")