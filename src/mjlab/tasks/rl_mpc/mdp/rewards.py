"""Reward helpers specific to residual foot placement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.rl_mpc.controller.backend import MpcGaitMode

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_placement_offset_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize unnecessary raw residual foothold commands."""
  return torch.sum(torch.square(env.action_manager.action), dim=1)


def foot_placement_lateral_offset_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize lateral residuals without reducing fore-aft terrain authority."""
  action = env.action_manager.action
  return torch.sum(torch.square(action[:, 1::2]), dim=1)


def stopped_joint_pose_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  action_name: str = "foot_placement",
  command_name: str = "twist",
  linear_threshold: float = 0.03,
  yaw_threshold: float = 0.05,
) -> torch.Tensor:
  """Squared joint error to the home pose, only for commanded stopping/stand."""
  robot = env.scene[asset_cfg.name]
  error = (
    robot.data.joint_pos[:, asset_cfg.joint_ids]
    - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  command = env.command_manager.get_command(command_name)
  if command is None:
    raise RuntimeError(f"Command {command_name!r} is unavailable")
  mode = getattr(env.action_manager.get_term(action_name), "gait_modes", None)
  if not isinstance(mode, torch.Tensor):
    raise TypeError("RL-MPC action must expose gait_modes")
  stopped = (mode == int(MpcGaitMode.STOPPING)) | (mode == int(MpcGaitMode.STAND))
  stopped &= torch.linalg.vector_norm(command[:, :2], dim=1) <= linear_threshold
  stopped &= command[:, 2].abs() <= yaw_threshold
  return error.square().sum(dim=1) * stopped


def straight_foot_placement_symmetry(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  action_name: str = "foot_placement",
) -> torch.Tensor:
  """Train bounded pair corrections; keep the existing log term name.

  This compares proposed residuals, never instantaneous measured foot positions
  on opposite trot phases. Each front/rear pair shares x and mirrors y.
  """
  action = env.action_manager.action.reshape(-1, 4, 2)
  command = env.command_manager.get_command(command_name)
  if command is None:
    raise RuntimeError(f"Command {command_name!r} is unavailable")
  mode = getattr(env.action_manager.get_term(action_name), "gait_modes", None)
  if not isinstance(mode, torch.Tensor):
    raise TypeError("RL-MPC action must expose gait_modes")
  straight = (command[:, 1].abs() <= 0.03) & (command[:, 2].abs() <= 0.05)
  left, right = action[:, (0, 2)], action[:, (1, 3)]
  # Convert the 5 mm X difference / 5 mm Y mirror-error allowance to actions.
  turning = (~straight).to(action.dtype)
  x_error = (left[..., 0] - right[..., 0]).abs()
  y_error = (left[..., 1] + right[..., 1]).abs()
  error = (x_error - turning[:, None] * (0.005 / 0.08)).clamp_min(0).square()
  error += (y_error - turning[:, None] * (0.005 / 0.03)).clamp_min(0).square()
  caps = action.new_tensor([0.015 / 0.08, 0.0075 / 0.03])
  excess = (action.abs() - caps).clamp_min(0).square().sum(dim=(1, 2))
  return (error.sum(dim=1) + turning * excess) * (mode == int(MpcGaitMode.TROT))
