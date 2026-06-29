# FILE: capture_gif.py
"""
LeKiwi pick-and-place demo — live viewer + growing plot + optional GIF capture.

Architecture:
  Base: LQR controller tracks waypoints in world frame [x, y, theta]
  Arm:  P-controller in EE space → fallback FK/IK → mirror joints to MuJoCo
  Gripper: Direct jaw position command

  The arm uses the fallback FK/IK path (not MuJoCo physics) because MuJoCo's
  position servos (kp=50) can't track small Jacobian steps under gravity.
  Joint positions are mirrored to MuJoCo for rendering.

Usage:
  python capture_gif.py              # live viewer + plot
  python capture_gif.py --gif        # render to GIF (headless)
  python capture_gif.py --gif --fast # render to GIF, fewer frames
"""
import os
import sys

# ── Parse args ──────────────────────────────────────────────────────────────
RENDER_GIF = "--gif" in sys.argv
FAST = "--fast" in sys.argv

# Set MuJoCo GL backend BEFORE importing mujoco
if RENDER_GIF:
    os.environ['MUJOCO_GL'] = 'egl'
    import matplotlib
    matplotlib.use('Agg')
else:
    import matplotlib
    matplotlib.use('TkAgg')

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import time
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUTPUT_PATH = str(HERE / "lekiwi_demo.gif")

from lekiwi_sim import LeKiwiSim
from lqr import LQR
from armrobot import ArmRobot


# ── EE Pose Trajectory ──────────────────────────────────────────────────────
EE_KEYFRAMES = [
    (50,  np.array([0.015,  0.101,  0.031, 0.0, 0.0, 0.0])),
    (100, np.array([0.015,  0.101, -0.050, 0.0, 0.0, 0.0])),
    (50,  np.array([0.015,  0.101, -0.050, 0.0, 0.0, 0.0])),
    (100, np.array([0.015,  0.101,  0.100, 0.0, 0.0, 0.0])),
    (200, np.array([0.015,  0.101,  0.100, 0.0, 0.0, 0.0])),
    (100, np.array([0.015,  0.101, -0.050, 0.0, 0.0, 0.0])),
    (50,  np.array([0.015,  0.101, -0.050, 0.0, 0.0, 0.0])),
    (100, np.array([0.015,  0.101,  0.031, 0.0, 0.0, 0.0])),
]

GRIP_SCHEDULE = [
    (150, 0.5),
    (450, 0.0),
]

BASE_WAYPOINTS = [
    (0.0, 0.0, 0.0),
    (0.4, 0.0, 0.0),
    (0.4, 0.0, 0.0),
    (0.4, 0.0, 0.0),
    (1.2, 0.0, 0.0),
    (1.2, 0.0, 0.0),
    (1.2, 0.0, 0.0),
]
BASE_WAYPOINT_STEPS = [50, 50, 150, 100, 200, 150, 100]

# ── Build schedules ─────────────────────────────────────────────────────────
total_steps = sum(k[0] for k in EE_KEYFRAMES)

base_schedule = []
for wp, n in zip(BASE_WAYPOINTS, BASE_WAYPOINT_STEPS):
    base_schedule.extend([np.array(wp)] * n)
base_schedule = np.array(base_schedule[:total_steps])

ee_schedule = []
for n, pose in EE_KEYFRAMES:
    ee_schedule.extend([pose.copy() for _ in range(n)])
ee_schedule = np.array(ee_schedule)


# ── Create sim ──────────────────────────────────────────────────────────────
sim = LeKiwiSim(dt=0.02)
sim.reset()

# ── Fallback arm (FK/IK, no MuJoCo physics) ─────────────────────────────────
NUM_DOF = 6
DT = 0.02
JOINT_LIMITS = np.array([
    [-3.0,   3.0],
    [-3.1416, 3.14],
    [-3.14,  3.1416],
    [-3.0,   3.14],
    [-3.1416, 3.1416],
    [-3.14,  3.0],
])
LINK_OFFSETS = np.array([
    [0.018300,  0.030600,  0.052200],
    [-0.001500, -0.114582,  0.018082],
    [-0.001500,  0.132932,  0.028720],
    [-0.020100,  0.025822, -0.055375],
    [0.019800,  0.026631, -0.013098],
    [0.0,        0.0,       0.0],
])
ROT_AXES = ["y", "z", "z", "x", "z", "z"]

arm_fallback = ArmRobot(NUM_DOF, DT, JOINT_LIMITS, LINK_OFFSETS, ROT_AXES)
T_home, _, _ = arm_fallback.forward_kinematics(np.zeros(6))
arm_fallback.state = np.array([T_home[0,3], T_home[1,3], T_home[2,3], 0.0, 0.0, 0.0])

# ── LQR for base ────────────────────────────────────────────────────────────
A_base = np.eye(3)
B_base = 0.02 * np.eye(3)
Q_base = np.diag([100.0, 100.0, 50.0])
R_base = np.diag([0.1, 0.1, 0.1])
base_ctrl = LQR(Q_base, R_base, A_base, B_base)

# ── Data logging ───────────────────────────────────────────────────────────
log_time = []
log_base_error = []
log_base_control = []
log_ee_pos = []

# ── Plot setup ─────────────────────────────────────────────────────────────
if RENDER_GIF:
    renderer = mujoco.Renderer(sim.engine.model, width=400, height=300)
    camera = mujoco.MjvCamera()
    camera.distance = 1.8
    camera.azimuth = 135
    camera.elevation = -20
    camera.lookat[:] = [0.0, 0.0, 0.1]
    frames = []
    viewer_ctx = None  # no viewer in headless mode
else:
    renderer = None
    camera = None
    frames = None
    viewer_ctx = "launch"  # will open viewer below

fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white', labelsize=7)
    ax.spines['bottom'].set_color('#555')
    ax.spines['top'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['right'].set_color('#555')

ax_err, ax_ctrl, ax_ee = axes
ax_err.set_ylabel('Base Error', color='white', fontsize=8)
ax_ctrl.set_ylabel('Base Control', color='white', fontsize=8)
ax_ee.set_ylabel('EE Position', color='white', fontsize=8)
ax_ee.set_xlabel('Time (s)', color='white', fontsize=8)

line_err_x, = ax_err.plot([], [], '#ff6b6b', lw=1, label='x err')
line_err_y, = ax_err.plot([], [], '#4ecdc4', lw=1, label='y err')
line_err_t, = ax_err.plot([], [], '#ffe66d', lw=1, label='θ err')
ax_err.legend(loc='upper right', fontsize=6, labelcolor='white')
ax_err.set_ylim(-0.5, 0.5)

line_ctrl_x, = ax_ctrl.plot([], [], '#ff6b6b', lw=1, label='vx')
line_ctrl_y, = ax_ctrl.plot([], [], '#4ecdc4', lw=1, label='vy')
line_ctrl_t, = ax_ctrl.plot([], [], '#ffe66d', lw=1, label='ω')
ax_ctrl.legend(loc='upper right', fontsize=6, labelcolor='white')
ax_ctrl.set_ylim(-0.6, 0.6)

line_ee_x, = ax_ee.plot([], [], '#ff6b6b', lw=1, label='x')
line_ee_y, = ax_ee.plot([], [], '#4ecdc4', lw=1, label='y')
line_ee_z, = ax_ee.plot([], [], '#45b7d1', lw=1, label='z')
ax_ee.legend(loc='upper right', fontsize=6, labelcolor='white')
ax_ee.set_ylim(-0.15, 0.25)

plt.tight_layout()
plt.ion()
plt.show(block=False)

# ── Simulation loop ─────────────────────────────────────────────────────────
capture_every = 4 if FAST else 2
grip_idx = 0

# Open MuJoCo viewer (only in live mode)
if not RENDER_GIF:
    viewer = mujoco.viewer.launch_passive(sim.engine.model, sim.engine.data)
    viewer.cam.distance = 1.8
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    viewer.cam.lookat[:] = [0.0, 0.0, 0.1]
else:
    viewer = None

for step in range(total_steps):
    t = step * 0.02

    # ── Arm: P-controller → fallback FK/IK ──
    target_ee = ee_schedule[step]
    current_ee = arm_fallback.get_state()
    vel = 2.0 * (target_ee - current_ee)
    vel = np.clip(vel, [-0.3, -0.3, -0.3, -1.0, -1.0, -1.0],
                        [0.3,  0.3,  0.3,  1.0,  1.0,  1.0])
    joints = arm_fallback.step(vel)

    # Mirror to MuJoCo
    sim.engine.set_arm_ctrl(joints)
    if grip_idx < len(GRIP_SCHEDULE) and step >= GRIP_SCHEDULE[grip_idx][0]:
        jaw_pos = GRIP_SCHEDULE[grip_idx][1]
        ctrl = sim.engine.data.ctrl.copy()
        ctrl[8] = jaw_pos
        sim.engine.set_full_ctrl(ctrl)
        grip_idx += 1

    # ── Base: LQR ──
    target_pose = base_schedule[step]
    current_pose = sim.base.get_state()
    base_vel = base_ctrl.compute(current_pose, target_pose)
    base_vel = np.clip(base_vel, [-0.5, -0.5, -1.0], [0.5, 0.5, 1.0])
    sim.base.step(base_vel)

    # ── Step physics ──
    sim.step()

    # ── Update viewer camera ──
    if viewer is not None:
        base_pose = sim.base.state
        viewer.cam.lookat[:] = [base_pose[0], base_pose[1], 0.1]

    # ── Log data ──
    log_time.append(t)
    base_err = target_pose - current_pose
    log_base_error.append(base_err.copy())
    log_base_control.append(base_vel.copy())
    log_ee_pos.append(arm_fallback.get_state()[:3].copy())

    # ── Update plot every 5 steps ──
    if step % 5 == 0 and step > 0:
        time_arr = np.array(log_time)
        err_arr = np.array(log_base_error)
        ctrl_arr = np.array(log_base_control)
        ee_arr = np.array(log_ee_pos)

        line_err_x.set_data(time_arr, err_arr[:, 0])
        line_err_y.set_data(time_arr, err_arr[:, 1])
        line_err_t.set_data(time_arr, err_arr[:, 2])
        ax_err.relim()
        ax_err.autoscale_view(scalex=False)

        line_ctrl_x.set_data(time_arr, ctrl_arr[:, 0])
        line_ctrl_y.set_data(time_arr, ctrl_arr[:, 1])
        line_ctrl_t.set_data(time_arr, ctrl_arr[:, 2])
        ax_ctrl.relim()
        ax_ctrl.autoscale_view(scalex=False)

        line_ee_x.set_data(time_arr, ee_arr[:, 0])
        line_ee_y.set_data(time_arr, ee_arr[:, 1])
        line_ee_z.set_data(time_arr, ee_arr[:, 2])
        ax_ee.relim()
        ax_ee.autoscale_view(scalex=False)

        fig.canvas.draw()
        if not RENDER_GIF:
            fig.canvas.flush_events()

    # ── Sync viewer ──
    if viewer is not None:
        viewer.sync()
        time.sleep(sim.engine.dt / 4)

        if not viewer.is_running():
            break

    # ── Capture frame for GIF ──
    if RENDER_GIF and step % capture_every == 0:
        renderer.update_scene(sim.engine.data, camera)
        frame = renderer.render()

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = plot_img[:, :, :3]

        plot_h = frame.shape[0]
        plot_w = int(plot_img.shape[1] * plot_h / plot_img.shape[0])
        y_ratio = plot_img.shape[0] / plot_h
        x_ratio = plot_img.shape[1] / plot_w
        y_idx = (np.arange(plot_h) * y_ratio).astype(int)
        x_idx = (np.arange(plot_w) * x_ratio).astype(int)
        plot_resized = plot_img[y_idx[:, None], x_idx]

        combined = np.hstack([frame, plot_resized])
        frames.append(combined)

if viewer is not None:
    viewer.close()

if RENDER_GIF and frames:
    import imageio.v3 as iio
    iio.imwrite(
        OUTPUT_PATH, frames,
        fps=50 // capture_every,
        loop=0,
        plugin='pillow',
        optimize=True,
    )
    print(f"✅ GIF saved: {OUTPUT_PATH}")
    print(f"   {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, {50 // capture_every} fps")
    file_size = Path(OUTPUT_PATH).stat().st_size
    print(f"   File size: {file_size / 1024:.0f} KB")

if renderer:
    renderer.close()
plt.ioff()
plt.close(fig)
print("✅ Demo complete")
