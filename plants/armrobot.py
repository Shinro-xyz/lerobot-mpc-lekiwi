import numpy as np
from typing import Optional, List
from components import Plant, PhysicsEngine
from factories.registry import register_plant


@register_plant("ArmRobot")
class ArmRobot(Plant):
    """6-DOF robotic arm plant with forward kinematics, Jacobian, and IK.

    Models a serial-link manipulator with configurable joint offsets and
    rotation axes. Supports two modes:
    1. **Standalone** — uses simplified FK/IK for quick testing
    2. **Physics engine** — attaches a PhysicsEngine for mesh-accurate Jacobian IK

    The arm operates in Cartesian space: step() takes a 6D velocity twist
    [dx, dy, dz, droll, dpitch, dyaw], integrates to a target pose, and
    uses inverse kinematics to compute joint angles.

    Usage:
        arm = ArmRobot(
            num_dof=6, dt=0.02,
            joint_limits=np.array([[-np.pi, np.pi]] * 6),
            joint_offsets=np.array([[...], [...], ...]),
            rot_axes=['y', 'z', 'z', 'x', 'z', 'z'],
            joint_names=['Rotation', 'Pitch', ...],
            ee_body_name='Moving_Jaw_08d-v1',
        )
        arm.physics_engine(physics_engine)
        joints = arm.step(np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]))
    """

    def __init__(
        self,
        num_dof: int,
        dt: float,
        joint_limits: np.ndarray,
        joint_offsets: np.ndarray,
        rot_axes: List[str],
        joint_names: Optional[List[str]] = None,
        ee_body_name: Optional[str] = None,
    ):
        self.num_dof = num_dof
        self.dt = dt
        self.state = np.zeros(6)
        self.joint_offsets = joint_offsets
        self.joint_limits = joint_limits
        self.axes = rot_axes
        self._last_joints = np.zeros(num_dof)
        self._engine = None
        self._ee_body_name = ee_body_name
        self._joint_names = joint_names if joint_names is not None else [f"joint_{i}" for i in range(num_dof)]

    def _get_ee_pos(self):
        return self._engine.get_body_xpos(self._ee_body_name)

    def _get_ee_jacobian(self):
        return self._engine.compute_jacobian_for_joints(self._ee_body_name, self._joint_names)

    def physics_engine(self, engine: Optional[PhysicsEngine]):
        self._engine = engine
        if engine is not None:
            if self._ee_body_name is None:
                self._ee_body_name = self._find_ee_body_name(engine)
            self._engine.forward()
            ee = self._get_ee_pos()
            self.state = np.array([ee[0], ee[1], ee[2], 0.0, 0.0, 0.0])
        else:
            T_home, _, _ = self.forward_kinematics(np.zeros(self.num_dof))
            self.state = np.array([T_home[0, 3], T_home[1, 3], T_home[2, 3], 0.0, 0.0, 0.0])

    def _find_ee_body_name(self, engine: PhysicsEngine) -> str:
        candidates = ["Moving_Jaw_08d-v1", "Moving_Jaw", "end_effector", "ee", "gripper"]
        for name in candidates:
            bid = engine.get_body_id(name)
            if bid >= 0:
                return name
        return engine.body_names[-1]

    def get_state(self):
        if self._engine is not None:
            ee = self._get_ee_pos()
            return np.array([ee[0], ee[1], ee[2], 0.0, 0.0, 0.0])
        return self.state.copy()

    def get_model(self):
        A = np.eye(6)
        B = self.dt * np.eye(6)
        return A, B

    def _pose_to_transform(self, pose: np.ndarray):
        x, y, z, roll, pitch, yaw = pose
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [x, y, z]
        return T

    def step(self, u: np.ndarray):
        if self._engine is not None:
            current_ee = self._get_ee_pos()
            target_ee = current_ee + u[:3] * self.dt
            joint_targets = self.engine_ik(target_ee)
            for name, val in zip(self._joint_names, joint_targets):
                self._engine.set_joint_ctrl(name, val)
            self._last_joints = np.array([self._engine.get_joint_qpos(n) for n in self._joint_names])
            ee = self._get_ee_pos()
            self.state = np.array([ee[0], ee[1], ee[2], 0.0, 0.0, 0.0])
            return self._last_joints

        self.state += self.dt * u
        target = self._pose_to_transform(self.state)
        q = self.inverse_kinematics(target)
        self._last_joints = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])
        return self._last_joints

    def _homogenous_transform(self, joint_angles: np.ndarray):
        sines, cosines = np.sin(joint_angles), np.cos(joint_angles)
        T = np.zeros((self.num_dof, 4, 4))
        for i in range(self.num_dof):
            axis = self.axes[i]
            s, c = sines[i], cosines[i]
            if axis == 'x':
                R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif axis == 'y':
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            elif axis == 'z':
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            else:
                raise ValueError(f"Invalid axis '{axis}' at joint {i}. Choose 'x', 'y', or 'z'.")
            offset_vector = self.joint_offsets[i, :3]
            T[i, :3, :3] = R
            T[i, :3, 3] = offset_vector
            T[i, 3, 3] = 1.0
        return T

    def forward_kinematics(self, joint_angles: np.ndarray):
        T_joints = self._homogenous_transform(joint_angles)
        T_cumulative = np.eye(4)
        positions = []
        axes = []
        for i in range(self.num_dof):
            T_cumulative = T_cumulative @ T_joints[i]
            axis_local = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[self.axes[i]]
            z_i = T_cumulative[:3, :3] @ axis_local
            positions.append(T_cumulative[:3, 3])
            axes.append(z_i)

        return T_cumulative, positions, axes

    def _jacobian(self, joint_angles: np.ndarray):
        T_endeffector, pos, axes = self.forward_kinematics(joint_angles)
        p_endeffector = pos[-1]
        J = np.zeros((6, self.num_dof))
        for i in range(self.num_dof):
            J[:3, i] = np.cross(axes[i], p_endeffector - pos[i])
            J[3:, i] = axes[i]
        return J

    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        max_iters: int = 100,
        q_init: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        max_step: float = 0.2,
    ):
        q = q_init if q_init is not None else self._last_joints.copy()
        for j in range(max_iters):
            T_cur, positions, axes = self.forward_kinematics(q)
            pos_err = target_pose[:3, 3] - T_cur[:3, 3]
            R_err = target_pose[:3, :3] @ T_cur[:3, :3].T
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))

            if angle < tol and np.linalg.norm(pos_err) < tol:
                break

            axis = np.array([R_err[2, 1] - R_err[1, 2],
                             R_err[0, 2] - R_err[2, 0],
                             R_err[1, 0] - R_err[0, 1]])

            if np.linalg.norm(axis) > 1e-6:
                ori_err = (axis / np.linalg.norm(axis)) * angle
            else:
                ori_err = np.zeros(3)

            v = np.concatenate([pos_err, ori_err])

            J = self._jacobian(q)
            dq = np.linalg.pinv(J) @ v
            dq = np.clip(dq, -max_step, max_step)
            q = q + dq
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

        return q

    def engine_ik(
        self,
        target_ee: np.ndarray,
        max_iters: int = 20,
        lam: float = 0.01,
        max_dq: float = 0.5,
    ):
        if self._engine is None:
            raise RuntimeError("engine_ik requires a physics engine (call physics_engine first)")

        current_ee = self._get_ee_pos()
        error = target_ee - current_ee

        if np.linalg.norm(error) < 0.001:
            return np.array([self._engine.get_joint_qpos(n) for n in self._joint_names])

        current_joints = np.array([self._engine.get_joint_qpos(n) for n in self._joint_names])

        for _ in range(max_iters):
            J = self._get_ee_jacobian()[:3, :]

            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(3), error)
            dq = np.clip(dq, -max_dq, max_dq)
            current_joints = current_joints + dq
            current_joints = np.clip(current_joints, self.joint_limits[:, 0], self.joint_limits[:, 1])

            for name, val in zip(self._joint_names, current_joints):
                self._engine.set_joint_qpos(name, val)
            self._engine.forward()

            current_ee = self._get_ee_pos()
            error = target_ee - current_ee
            if np.linalg.norm(error) < 0.001:
                break

        for name, val in zip(self._joint_names, current_joints):
            self._engine.set_joint_qpos(name, val)
        self._engine.forward()

        return current_joints

    @classmethod
    def from_config(cls, config):
        joint_names = config["joint_groups"][config["joint_group"]]
        engine = config["engine"]
        limits = np.array([engine.get_joint_limits(n) for n in joint_names])
        num_dof = config["num_dof"]
        plant = cls(
            num_dof=num_dof,
            dt=config["dt"],
            joint_limits=limits,
            joint_offsets=np.array(config["joint_offsets"]),
            rot_axes=config["rot_axes"],
            joint_names=joint_names,
            ee_body_name=config.get("ee_body_name"),
        )
        plant.physics_engine(engine)
        return plant
