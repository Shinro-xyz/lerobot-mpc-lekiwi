# FILE: lekiwi_sim.py (simulation orchestration)
"""
Simulation orchestration for the LeKiwi robot.

Provides LeKiwiSim (convenience wrapper), RobotSim (generic YAML factory),
run_simulation (closed-loop runner), and interactive_viewer.

The MuJoCoEngine lives in physics_engine/mujoco.py.
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from typing import Optional, Callable, Any
import yaml

from components import Controller
from physics_engine.mujoco import MuJoCoEngine

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
MJCF_PATH = str(HERE / "lekiwi-sim" / "mjcf_lcmm_robot.xml")


# ── LeKiwiSim (convenience wrapper) ───────────────────────────────────────
class LeKiwiSim:
    """
    High-level wrapper that connects ArmRobot + HolonomicMobileRobot
    to a shared MuJoCoEngine.

    Usage:
        sim = LeKiwiSim()
        sim.arm.step(target_joints)       # MuJoCo-backed arm
        sim.base.step(target_velocity)     # MuJoCo-backed base
        sim.engine.step()                 # advance physics
        arm_state = sim.arm.get_state()   # read from MuJoCo
    """

    ARM_JOINT_NAMES   = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    DRIVE_JOINT_NAMES = [
        "ST3215_Servo_Motor-v1-2_Hub---Servo",
        "ST3215_Servo_Motor-v1-1_Hub-2---Servo",
        "ST3215_Servo_Motor-v1_Revolute-40",
    ]

    def __init__(self, dt: float = 0.02, xml_string: str = None, assets: dict = None):
        from plants.armrobot import ArmRobot
        from plants.holonomicmobilerobot import HolonomicMobileRobot

        self.engine = MuJoCoEngine(dt=dt, xml_string=xml_string, assets=assets)

        # Arm: 6-DOF, limits from MuJoCo, offsets and axes from the MJCF model
        rot_axes = ["y", "z", "z", "x", "z", "z"]

        link_offsets = np.array([
            [0.018300,  0.030600,  0.052200],
            [-0.001500, -0.114582,  0.018082],
            [-0.001500,  0.132932,  0.028720],
            [-0.020100,  0.025822, -0.055375],
            [0.019800,  0.026631, -0.013098],
            [0.0,        0.0,       0.0],
        ])

        arm_limits = np.array([self.engine.get_joint_limits(n) for n in self.ARM_JOINT_NAMES])

        self.arm = ArmRobot(
            num_dof=6,
            dt=dt,
            joint_limits=arm_limits,
            joint_offsets=link_offsets,
            rot_axes=rot_axes,
            joint_names=self.ARM_JOINT_NAMES,
            ee_body_name="Moving_Jaw_08d-v1",
        )
        self.arm.physics_engine(self.engine)

        self.base = HolonomicMobileRobot(
            num_wheels=3,
            radius_robots=0.12,
            gamma=-np.pi / 2,
            radius_wheels=0.09,
            dt=dt,
        )
        self.base.physics_engine(self.engine)

    def reset(self):
        self.engine.reset()
        self.base.state = np.zeros(3, dtype=np.float64)

    def step(self):
        self.engine.step()
        base_state = self.base.state
        self.engine.data.qpos[0] = base_state[0]
        self.engine.data.qpos[1] = base_state[1]
        self.engine.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.engine.data.qvel[:6] = 0.0
        if hasattr(self.base, '_target_wheel_delta') and self.base._target_wheel_delta is not None:
            for name, delta in zip(self.DRIVE_JOINT_NAMES, self.base._target_wheel_delta):
                jid = self.engine._joint_name_to_id[name]
                qpos_adr = self.engine.model.jnt_qposadr[jid]
                self.engine.data.qpos[qpos_adr] += delta
            self.base._target_wheel_delta = None

    def get_state(self) -> dict:
        return self.engine.get_sensor_data()


# ── RobotSim (generic factory from YAML config) ──────────────────────────
class RobotSim:
    """
    Generic robot simulation factory. Reads a YAML config and creates the
    engine, plants, and wiring automatically.

    Usage:
        sim = RobotSim("robot_config.yaml")
        sim.arm.step(twist)
        sim.base.step(velocity)
        sim.step()
        state = sim.get_state()
    """

    def __init__(self, config_path: str, xml_string: str = None, assets: dict = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        dt = self.config.get("dt", 0.02)
        model_path = self.config.get("model", "")

        if xml_string is not None:
            self.engine = MuJoCoEngine(dt=dt, xml_string=xml_string, assets=assets)
        else:
            self.engine = MuJoCoEngine(model_path=model_path, dt=dt)

        # Build joint group name→name-list mapping
        joint_groups = self.config.get("joint_groups", {})

        # Instantiate each plant from config via registry
        self._plants = {}
        for plant_cfg in self.config.get("plants", []):
            ptype = plant_cfg["type"]
            pname = plant_cfg["name"]

            from factories.registry import _PLANT_REGISTRY
            cls = _PLANT_REGISTRY[ptype]
            plant_config = {**plant_cfg, "joint_groups": joint_groups, "engine": self.engine, "dt": dt}
            plant = cls.from_config(plant_config)

            self._plants[pname] = plant
            setattr(self, pname, plant)

    def reset(self):
        self.engine.reset()
        for name, plant in self._plants.items():
            if hasattr(plant, 'state') and isinstance(plant.state, np.ndarray):
                plant.state = np.zeros_like(plant.state)

    def step(self):
        self.engine.step()
        # LeKiwi-specific: teleport free joint to match kinematic base state
        if hasattr(self, 'base') and hasattr(self.engine, 'has_free_joint') and self.engine.has_free_joint:
            base_state = self.base.state
            self.engine.data.qpos[0] = base_state[0]
            self.engine.data.qpos[1] = base_state[1]
            self.engine.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.engine.data.qvel[:6] = 0.0
            if hasattr(self.base, '_target_wheel_delta') and self.base._target_wheel_delta is not None:
                drive_joints = self.config.get("joint_groups", {}).get("drive_joints", [])
                for name, delta in zip(drive_joints, self.base._target_wheel_delta):
                    jid = self.engine._joint_name_to_id[name]
                    qpos_adr = self.engine.model.jnt_qposadr[jid]
                    self.engine.data.qpos[qpos_adr] += delta
                self.base._target_wheel_delta = None

    def get_state(self) -> dict:
        return self.engine.get_sensor_data()

    def get_plant(self, name: str) -> Any:
        return self._plants.get(name)


def run_simulation(
    sim,
    controller: Controller,
    max_steps: int = 1000,
    render: bool = True,
    callback: Optional[Callable] = None,
):
    """
    Run a closed-loop simulation with optional viewer.

    The controller receives the full qpos vector and returns a full ctrl vector.
    `sim` must have `.engine` (MuJoCoEngine), `.reset()`, and `.step()`.
    """
    sim.reset()

    if render:
        with mujoco.viewer.launch_passive(sim.engine.model, sim.engine.data) as viewer:
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -30

            for step in range(max_steps):
                state = sim.engine.get_full_qpos()
                action = controller.compute(state)
                sim.engine.set_full_ctrl(action)
                sim.engine.step()

                if callback:
                    callback(step, sim)

                viewer.sync()
                import time
                time.sleep(sim.engine.dt / 4)

                if not viewer.is_running():
                    break
    else:
        for step in range(max_steps):
            state = sim.engine.get_full_qpos()
            action = controller.compute(state)
            sim.engine.set_full_ctrl(action)
            sim.engine.step()

            if callback:
                callback(step, sim)


# ── Interactive Viewer ────────────────────────────────────────────────────
def interactive_viewer(model_path: str = MJCF_PATH):
    """Open an interactive MuJoCo viewer for manual inspection."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Interactive viewer opened. Close window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--viewer" in sys.argv:
        interactive_viewer()
    else:
        # Smoke test
        engine = MuJoCoEngine()
        engine.print_info()

        ctrl = np.zeros(engine.model.nu)
        ctrl[3] = 0.5
        ctrl[4] = -0.3
        ctrl[5] = 0.8

        for _ in range(50):
            engine.set_full_ctrl(ctrl)
            engine.step()

        print(f"\nAfter 50 steps with arm movement:")
        print(f"  Arm joints: {engine.get_arm_qpos(LeKiwiSim.ARM_JOINT_NAMES)}")
        print(f"  Base pose:  {engine.get_base_pose()}")
        print("✅ MuJoCoEngine smoke test passed")
