"""MCP server for shinro-python-modules — controller and estimator tools."""

import json
from mcp.server.fastmcp import FastMCP
import numpy as np
from typing import Any

from factories.registry import _CONTROLLER_REGISTRY, _ESTIMATOR_REGISTRY
from factories.controller_factory import ControllerFactory
from factories.estimator_factory import EstimatorFactory
from utils.array_backend import NumpyBackend

server = FastMCP("shinro")
_store: dict[str, Any] = {}


def _to_list(arr: np.ndarray) -> list[float]:
    return arr.flatten().tolist()


def _from_list(data: list[float]) -> np.ndarray:
    return np.array(data, dtype=np.float64)


def _to_col(data: list[float]) -> np.ndarray:
    """Convert a flat list to a column vector (n, 1)."""
    return np.array(data, dtype=np.float64).reshape(-1, 1)


def _from_col(arr: np.ndarray) -> list[float]:
    """Convert a column vector (n, 1) to a flat list."""
    return arr.flatten().tolist()


def _set_mpc_default_constraints(ctrl: Any) -> None:
    """Set unconstrained bounds on an MPC controller if none were set."""
    if not hasattr(ctrl, "A_constraints"):
        import scipy.sparse as sp
        n_c = ctrl.m
        F = np.eye(n_c)
        lo = np.full((n_c,), -1e10)
        hi = np.full((n_c,), 1e10)
        ctrl.A_constraints = sp.csc_matrix(
            sp.block_diag([sp.coo_array(F)] * ctrl.N)
        )
        ctrl.lcons = np.tile(lo, ctrl.N)
        ctrl.ucons = np.tile(hi, ctrl.N)


# ── Controller tools ──────────────────────────────────────────────────────────


@server.tool()
def create_controller(
    name: str,
    config_path: str | None = None,
    type: str | None = None,
    params: dict | None = None,
) -> str:
    """Create a controller from a TOML config file or inline parameters.

    Args:
        name: Unique name to store this controller instance.
        config_path: Path to a TOML config file (e.g. configs/controllers/lqr_base.toml).
        type: Controller type when using inline params (PID, LQR, MPC_DeltaU, MPC_LTI).
        params: Inline parameters (same keys as TOML config, e.g. dt, kp, ki, kd).
    """
    if config_path:
        factory = ControllerFactory(config_path)
        ctrl = factory.create(backend=NumpyBackend())
    elif type:
        if type not in _CONTROLLER_REGISTRY:
            available = list(_CONTROLLER_REGISTRY.keys())
            return f"Unknown controller type '{type}'. Available: {available}"
        if not params:
            return f"Provide 'params' for controller type '{type}'"
        cls = _CONTROLLER_REGISTRY[type]
        ctrl = cls.from_config(params, backend=NumpyBackend())
    else:
        return "Provide either 'config_path' or both 'type' and 'params'"

    registry_name = getattr(ctrl, "_registry_name", "")
    if registry_name in ("MPC_LTI", "MPC_DeltaU"):
        _set_mpc_default_constraints(ctrl)

    _store[name] = ctrl
    return f"Created {type or 'config-based'} controller '{name}'"


@server.tool()
def controller_compute(
    name: str,
    state: list[float],
    reference: list[float] | None = None,
    u_prev: list[float] | None = None,
) -> str:
    """Compute a control action from a named controller.

    Args:
        name: Name of the stored controller instance.
        state: Current state vector.
        reference: Target/reference state vector (defaults to zeros).
        u_prev: Previous control input (required for MPC_DeltaU, ignored otherwise).
    """
    if name not in _store:
        return f"No controller named '{name}'. Create one first."

    ctrl = _store[name]
    state_arr = _from_list(state)
    ref_arr = _from_list(reference) if reference is not None else np.zeros_like(state_arr)
    registry_name = getattr(ctrl, "_registry_name", None)

    try:
        if registry_name == "MPC_DeltaU":
            u_prev_arr = _from_list(u_prev) if u_prev is not None else np.zeros(ctrl.m)
            action = ctrl.compute(state_arr, u_prev=u_prev_arr)
        elif registry_name == "MPC_LTI":
            action = ctrl.compute(state_arr)
        else:
            action = ctrl.compute(state_arr, ref_arr)
    except Exception as e:
        return f"Error in compute: {e}"

    return json.dumps({"action": _to_list(action)})


@server.tool()
def controller_reset(name: str) -> str:
    """Reset a named controller's internal state.

    Args:
        name: Name of the stored controller instance.
    """
    if name not in _store:
        return f"No controller named '{name}'"
    _store[name].reset()
    return f"Controller '{name}' reset"


@server.tool()
def list_controller_types() -> str:
    """List all registered controller types available for creation."""
    return json.dumps({"controller_types": list(_CONTROLLER_REGISTRY.keys())})


@server.tool()
def list_controllers() -> str:
    """List all created controller instances."""
    info = {}
    for name, ctrl in _store.items():
        registry_name = getattr(ctrl, "_registry_name", type(ctrl).__name__)
        info[name] = registry_name
    return json.dumps({"controllers": info})


# ── Estimator tools ──────────────────────────────────────────────────────────


@server.tool()
def create_estimator(
    name: str,
    config_path: str | None = None,
    type: str | None = None,
    params: dict | None = None,
) -> str:
    """Create a state estimator from a TOML config file or inline parameters.

    Args:
        name: Unique name to store this estimator instance.
        config_path: Path to a TOML config file (e.g. configs/estimators/luenberger_base.toml).
        type: Estimator type when using inline params (KalmanFilter, LuenbergerObserver).
        params: Inline parameters (same keys as TOML config, e.g. dt, process_noise, measurement_noise).
    """
    if config_path:
        factory = EstimatorFactory(config_path)
        est = factory.create(backend=NumpyBackend())
    elif type:
        if type not in _ESTIMATOR_REGISTRY:
            available = list(_ESTIMATOR_REGISTRY.keys())
            return f"Unknown estimator type '{type}'. Available: {available}"
        if not params:
            return f"Provide 'params' for estimator type '{type}'"
        cls = _ESTIMATOR_REGISTRY[type]
        est = cls.from_config(params, backend=NumpyBackend())
    else:
        return "Provide either 'config_path' or both 'type' and 'params'"

    _store[name] = est
    return f"Created {type or 'config-based'} estimator '{name}'"


@server.tool()
def estimator_estimate(
    name: str,
    measurement: list[float],
    control_input: list[float],
) -> str:
    """Run one predict-update cycle of a named estimator.

    Both estimators use column vectors internally. The server converts
    flat lists to (n, 1) columns and back.

    Args:
        name: Name of the stored estimator instance.
        measurement: Observation vector from sensors.
        control_input: Control vector applied at this step.
    """
    if name not in _store:
        return f"No estimator named '{name}'. Create one first."

    est = _store[name]
    meas_col = _to_col(measurement)
    ctrl_col = _to_col(control_input)

    try:
        x_hat = est.estimate(meas_col, ctrl_col)
    except Exception as e:
        return f"Error in estimate: {e}"

    return json.dumps({"state_estimate": _from_col(x_hat)})


@server.tool()
def estimator_reset(name: str) -> str:
    """Reset a named estimator's internal state.

    Args:
        name: Name of the stored estimator instance.
    """
    if name not in _store:
        return f"No estimator named '{name}'"
    _store[name].reset()
    return f"Estimator '{name}' reset"


@server.tool()
def list_estimator_types() -> str:
    """List all registered estimator types available for creation."""
    return json.dumps({"estimator_types": list(_ESTIMATOR_REGISTRY.keys())})


@server.tool()
def list_estimators() -> str:
    """List all created estimator instances."""
    info = {}
    for name, ctrl in _store.items():
        registry_name = getattr(ctrl, "_registry_name", type(ctrl).__name__)
        info[name] = registry_name
    return json.dumps({"estimators": info})


if __name__ == "__main__":
    server.run(transport="stdio")
