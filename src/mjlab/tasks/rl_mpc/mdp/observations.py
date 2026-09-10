"""Observations specific to the hybrid RL-MPC controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.tasks.rl_mpc.controller.backend import NUM_MPC_GAIT_MODES

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def mpc_action_history(
  env: ManagerBasedRlEnv, action_name: str = "foot_placement"
) -> torch.Tensor:
  """Return a[t-1], a[t-2], a[t-3], a[t-4], clipped and zero padded."""
  history = getattr(env.action_manager.get_term(action_name), "action_history", None)
  if not isinstance(history, torch.Tensor) or history.shape != (env.num_envs, 32):
    raise ValueError("RL-MPC action history must have shape (num_envs, 32)")
  return history


def mpc_state_history(
  env: ManagerBasedRlEnv, action_name: str = "foot_placement"
) -> torch.Tensor:
  """Reference-style complete state history, including gait mode per frame."""
  history = getattr(env.action_manager.get_term(action_name), "state_history", None)
  if not isinstance(history, torch.Tensor) or history.shape != (env.num_envs, 375):
    raise ValueError("MPC state history must have shape (num_envs, 375)")
  return history


def mpc_gait_mode(
  env: ManagerBasedRlEnv,
  action_name: str = "foot_placement",
) -> torch.Tensor:
  """Return the actual TROT/STOPPING/STAND controller mode as one-hot."""
  action_term = env.action_manager.get_term(action_name)
  modes = getattr(action_term, "gait_modes", None)
  if not isinstance(modes, torch.Tensor):
    raise TypeError(f"Action term {action_name!r} does not expose gait_modes")
  if modes.shape != (env.num_envs,):
    raise ValueError(
      f"gait_modes must have shape ({env.num_envs},), got {tuple(modes.shape)}"
    )
  if modes.dtype == torch.bool or modes.is_floating_point():
    raise TypeError("gait_modes must use an integer tensor dtype")
  if torch.any((modes < 0) | (modes >= NUM_MPC_GAIT_MODES)):
    raise ValueError("gait_modes contains an invalid controller mode")
  return F.one_hot(modes.to(torch.long), num_classes=NUM_MPC_GAIT_MODES).to(
    dtype=torch.float32
  )
