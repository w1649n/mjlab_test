"""Compare phase-aligned touchdown and two-step stopping, without training."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from export_rl_mpc_policy import load_actor
from validate_rl_mpc import _set_exact_manual_command

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.rl_mpc.config.g23.env_cfgs import (
  syncai_g23_pure_mpc_validation_env_cfg,
)

torch.set_num_threads(1)
cfg = syncai_g23_pure_mpc_validation_env_cfg()
cfg.scene.num_envs = 1
cfg.auto_reset = False
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
checkpoint = (
  ROOT
  / "logs/rsl_rl/g23_rl_mpc_foot_state_history_v3/2026-09-09_15-00-51/model_9999.pt"
)
actor = load_actor(checkpoint)[0].as_onnx(verbose=False).eval()
results = []
try:
  for variant in ("pure_mpc", "old_policy_controller", "new_policy_controller"):
    env.reset(seed=42)
    term = env.action_manager.get_term("foot_placement")
    controller = term._controllers[0]._runner.cMPC
    controller.symmetric_residuals = variant == "new_policy_controller"
    controller.stop_reposition_steps = 2 if variant == "new_policy_controller" else 0
    command_term = env.command_manager.get_term("twist")
    feet_x = [[] for _ in range(4)]
    offsets_x = [[] for _ in range(4)]
    previous_swing = np.zeros(4, dtype=bool)
    step_launches = []
    prior_remaining = None
    poses = []
    modes = []
    terminated = False
    for phase, duration, vx in (("stand", 2, 0.0), ("walk", 6, 0.2), ("stop", 4, 0.0)):
      _set_exact_manual_command(command_term, 0, (vx, 0.0, 0.0))
      for tick in range(round(duration / env.step_dt)):
        obs = term.state_history.clone()
        obs[:, 61] = obs[:, 2].cos() * vx
        obs[:, 62] = obs[:, 2].sin() * vx
        obs[:, 63] = 0
        term._observation_history.frames[:, 0, 61:64] = obs[:, 61:64]
        with torch.inference_mode():
          action = torch.zeros((1, 8)) if variant == "pure_mpc" else actor(obs)
        _, _, done, timeout, _ = env.step(action.clamp(-1, 1))
        swing = controller.last_gait_inputs["swing_states"].reshape(4) > 0
        if phase == "walk" and tick * env.step_dt >= 2:
          for leg in np.flatnonzero(previous_swing & ~swing):
            feet_x[leg].append(float(controller.foot_positions[leg, 0, 0]))
            offsets_x[leg].append(
              float(controller._latched_foot_placement_offsets[leg, 0])
            )
        previous_swing = swing.copy()
        if phase == "stop":
          remaining = controller._stop_steps_remaining
          if controller._stop_step_active and remaining != prior_remaining:
            step_launches.append(
              np.flatnonzero(controller._stopping_swing_legs).tolist()
            )
            prior_remaining = remaining
          if tick * env.step_dt >= 2:
            robot = env.scene["robot"].data
            poses.append(
              float((robot.joint_pos - robot.default_joint_pos).square().mean())
            )
            modes.append(int(controller.gait_mode))
        if bool(done.any() or timeout.any()):
          terminated = True
          break
      if terminated:
        break
    mean_x = [float(np.mean(x)) if x else None for x in feet_x]
    results.append(
      {
        "variant": variant,
        "touchdown_samples": [len(x) for x in feet_x],
        "mean_touchdown_x_body_m": mean_x,
        "rear_touchdown_x_difference_m": None
        if mean_x[2] is None or mean_x[3] is None
        else mean_x[3] - mean_x[2],
        "mean_touchdown_residual_x_m": [
          float(np.mean(x)) if x else None for x in offsets_x
        ],
        "stop_diagonal_launches": step_launches,
        "stop_pose_rmse_rad": float(np.sqrt(np.mean(poses))) if poses else None,
        "stop_tail_modes": sorted(set(modes)),
        "terminated": terminated,
      }
    )
    (Path(__file__).parent / "phase_aligned_validation.json").write_text(
      json.dumps(results, indent=2)
    )
    print(json.dumps(results[-1]), flush=True)
finally:
  env.close()
