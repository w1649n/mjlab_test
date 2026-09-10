"""Termination terms for RL-MPC terrain-input failures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_scan_failed(
  env: ManagerBasedRlEnv,
  action_name: str = "foot_placement",
) -> torch.Tensor:
  """Terminate only environments whose scan was unsafe for MPC this step."""
  action_term = env.action_manager.get_term(action_name)
  value = getattr(action_term, "terrain_scan_failed", None)
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Action term {action_name!r} does not expose terrain_scan_failed")
  if value.shape != (env.num_envs,):
    raise ValueError(
      f"terrain_scan_failed must have shape ({env.num_envs},), got {tuple(value.shape)}"
    )
  if value.dtype != torch.bool:
    raise TypeError("terrain_scan_failed must be a boolean tensor")
  return value
