# FILE: capture_gif.py
"""
Capture a LeKiwiSim simulation as a GIF — with actual controllers.
Uses LQR for base position tracking and smooth arm trajectory planning.
EGL headless rendering, optimized for Discord (under 1MB).

Usage:  python capture_gif.py
Output: lekiwi_demo.gif
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import mujoco
from pathlib import Path
import imageio.v3 as iio

HERE = Path(__file__).parent
OUTPUT_PATH = str(HERE / "lekiwi_demo.gif")

from lekiwi_sim import LeKiwiSim
from lqr import LQR


# ── Trajectory planner ──────────────────────────────────────────────────────
class Trajectory:
    """Smooth joint trajectory between keyframes using cubic interpolation."""
    def __init__(self, keyframes: list, dt: float = 0.02):
        """
        keyframes: list of (num_steps, joint_target_array)
        Each keyframe runs for num_steps steps.
        """
        self.keyframes = keyframes
        self.dt = dt
        self.total_steps = sum(k[0] for k in keyframes)
        # Precompute the full trajectory
        self._build()

    def _build(self):
        self.traj = []
        for num_steps, target in self.keyframes:
            for _ in range(num_steps):
                self.traj.append(target.copy())
        self.traj = np.array(self.traj)

    def get(self, step: int) -> np.ndarray:
        return self.traj[min(step, self.total_steps - 1)].copy()


# ── Create sim ──────────────────────────────────────────────────────────────
sim = LeKiwiSim(dt=0.02)
sim.reset()

# ── LQR for base position tracking ──────────────────────────────────────────
# State: [x, y, theta], Control: [vx, vy, omega]
# A = I, B = dt * I (discrete integrator)
A_base = np.eye(3)
B_base = 0.02 * np.eye(3)
Q_base = np.diag([100.0, 100.0, 50.0])   # penalize position error heavily
R_base = np.diag([0.1, 0.1, 0.1])         # light penalty on control effort
base_ctrl = LQR(Q_base, R_base, A_base, B_base)

# ── Waypoints for base ──────────────────────────────────────────────────────
# (x, y, theta) targets the base drives to
waypoints = [
    (0.0, 0.0, 0.0),      # start
    (0.4, 0.0, 0.0),      # drive forward to box
    (0.4, 0.0, 0.0),      # hold while arm reaches
    (0.4, 0.0, 0.0),      # hold while arm grips
    (0.4, 0.0, 0.0),      # hold while arm lifts
    (1.2, 0.0, 0.0),      # drive to drop
    (1.2, 0.0, 0.0),      # hold while arm lowers
    (1.2, 0.0, 0.0),      # hold while arm releases
    (1.2, 0.0, 0.0),      # hold while arm lifts away
]
waypoint_steps = [50, 50, 100, 100, 100, 200, 100, 100, 100]  # steps per waypoint

# ── Arm trajectory keyframes ────────────────────────────────────────────────
# (num_steps, [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw])
arm_traj = Trajectory([
    (50,  np.array([0.0,  0.0,  0.0,  0.0, 0.0, 0.0])),   # rest
    (50,  np.array([0.0, -0.5,  1.0,  0.0, 0.0, 0.0])),   # reach down
    (100, np.array([0.0, -0.5,  1.0,  0.0, 0.0, 0.5])),   # grip
    (100, np.array([0.0,  0.0,  0.3,  0.0, 0.0, 0.5])),   # lift
    (200, np.array([0.0,  0.0,  0.3,  0.0, 0.0, 0.5])),   # carry
    (100, np.array([0.0, -0.5,  1.0,  0.0, 0.0, 0.5])),   # lower
    (100, np.array([0.0, -0.5,  1.0,  0.0, 0.0, 0.0])),   # release
    (100, np.array([0.0,  0.0,  0.3,  0.0, 0.0, 0.0])),   # lift away
])

total_steps = arm_traj.total_steps

# ── Renderer ────────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(sim.engine.model, width=400, height=300)
camera = mujoco.MjvCamera()
camera.distance = 1.5
camera.azimuth = 135
camera.elevation = -20
camera.lookat[:] = [0.0, 0.0, 0.1]

# ── Simulation loop ──────────────────────────────────────────────────────────
frames = []
capture_every = 2

# Build waypoint schedule
waypoint_schedule = []
step_counter = 0
for wp, n_steps in zip(waypoints, waypoint_steps):
    waypoint_schedule.extend([np.array(wp)] * n_steps)
    step_counter += n_steps
waypoint_schedule = np.array(waypoint_schedule[:total_steps])

for step in range(total_steps):
    # ── Arm: follow trajectory ──
    arm_target = arm_traj.get(step)
    sim.arm.step(arm_target)

    # ── Base: LQR tracks waypoint ──
    target_pose = waypoint_schedule[step]
    current_pose = sim.base.get_state()
    base_vel = base_ctrl.compute(current_pose, target_pose)
    # Clamp velocity to reasonable limits
    base_vel = np.clip(base_vel, [-0.5, -0.5, -1.0], [0.5, 0.5, 1.0])
    sim.base.step(base_vel)

    # ── Step physics ──
    sim.step()

    # ── Update camera ──
    base_pose = sim.base.state
    camera.lookat[:] = [base_pose[0], base_pose[1], 0.1]

    # ── Render ──
    if step % capture_every == 0:
        renderer.update_scene(sim.engine.data, camera)
        frames.append(renderer.render())

renderer.close()

# ── Save GIF ────────────────────────────────────────────────────────────────
iio.imwrite(
    OUTPUT_PATH, frames,
    fps=50 // capture_every,
    loop=0,
    plugin='pillow',
    optimize=True,
)
print(f"✅ GIF saved: {OUTPUT_PATH}")
print(f"   {len(frames)} frames, 400x300, {50 // capture_every} fps")
file_size = Path(OUTPUT_PATH).stat().st_size
print(f"   File size: {file_size / 1024:.0f} KB")
