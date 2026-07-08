# FILE: controllers/lerobot_adapter.py (70 lines)
"""
LeRobot policy adapter — wraps any Hugging Face LeRobot policy as a Controller.

Allows swapping between classical control (LQR, MPC) and learned policies
(diffusion, ACT, pi0, smolvla, etc.) with the same --controller flag.

Usage:
    # In configs/controllers/lerobot_diffusion.toml:
    #   type = "lerobot_diffusion"
    #   policy_type = "diffusion"
    #   checkpoint = "huggingface/lerobot-diffusion-lekiwi"
    #   use_camera = false
    #
    # python -m demos.demo_base_tracking --controller lerobot_diffusion
"""

import numpy as np
from typing import Optional
from components import Controller
from factories.registry import register_controller


@register_controller("lerobot_diffusion")
class LeRobotDiffusionAdapter(Controller):
    """Wrap a LeRobot diffusion policy as a Controller.

    The policy is loaded from a Hugging Face checkpoint or local path.
    The `compute()` method converts the plant state into LeRobot's
    observation dict format, runs the policy, and returns the action.

    When no camera is available, the observation dict contains only
    state/position data. When a camera is connected, the adapter
    accepts image frames via `update_camera()`.
    """

    def __init__(self, policy, use_camera: bool = False, device: str = "cpu"):
        self.policy = policy
        self.use_camera = use_camera
        self.device = device
        self._latest_camera_frame: Optional[np.ndarray] = None

    def update_camera(self, frame: np.ndarray):
        """Feed a camera frame for the next policy step."""
        self._latest_camera_frame = frame

    def compute(self, state: np.ndarray, target: Optional[np.ndarray] = None) -> np.ndarray:
        """Run the LeRobot policy on the current state.

        Args:
            state: Plant state vector (n_x,). For the arm, this is
                   joint positions. For the base, this is [x, y, theta].
            target: Ignored for learned policies — they generate actions
                    from observation alone.

        Returns:
            Action vector (n_u,) that the plant can consume.
        """
        import torch

        # Build LeRobot observation dict
        obs = {"observation.state": torch.from_numpy(state).float().unsqueeze(0)}

        if self.use_camera and self._latest_camera_frame is not None:
            # Convert HWC numpy to CHW tensor and add batch dim
            frame_tensor = torch.from_numpy(self._latest_camera_frame).float()
            if frame_tensor.ndim == 3:  # HWC
                frame_tensor = frame_tensor.permute(2, 0, 1)  # CHW
            obs["observation.images.cam"] = frame_tensor.unsqueeze(0)
            self._latest_camera_frame = None

        # Move to device and run policy
        obs = {k: v.to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            action = self.policy.select_action(obs)

        # Return as numpy array, squeezing batch dim
        return action.squeeze(0).cpu().numpy()

    def reset(self):
        """Reset the policy's internal state."""
        self.policy.reset()
        self._latest_camera_frame = None

    @classmethod
    def from_config(cls, config):
        """Load a LeRobot policy from a config dict.

        Config fields:
            policy_type:   Type of policy ("diffusion", "act", "pi0", etc.)
            checkpoint:    Hugging Face repo ID or local path
            use_camera:    Whether to expect camera observations (default: false)
            device:        Torch device (default: "cpu")
        """
        policy_type = config["policy_type"]
        checkpoint = config["checkpoint"]
        use_camera = config.get("use_camera", False)
        device = config.get("device", "cpu")

        from lerobot.policies import make_policy
        policy = make_policy(policy_type, pretrained_path=checkpoint)
        policy.to(device)
        policy.eval()

        return cls(policy, use_camera=use_camera, device=device)