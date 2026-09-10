"""Training metrics for residual foot-placement locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def mean_abs_lateral_foot_placement_action(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  """Return the mean absolute normalized dy action for each environment."""
  return torch.mean(torch.abs(env.action_manager.action[:, 1::2]), dim=1)


def max_abs_hip_x(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the largest absolute HipX angle per environment, in radians."""
  asset = env.scene[asset_cfg.name]
  return torch.amax(torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)


def terrain_scan_miss_fraction(
  env: ManagerBasedRlEnv,
  action_name: str = "foot_placement",
) -> torch.Tensor:
  """Return the latest missed-ray fraction for each environment."""
  action_term = env.action_manager.get_term(action_name)
  value = getattr(action_term, "terrain_scan_miss_fraction", None)
  if not isinstance(value, torch.Tensor):
    raise TypeError(
      f"Action term {action_name!r} does not expose terrain_scan_miss_fraction"
    )
  if value.shape != (env.num_envs,):
    raise ValueError(
      "terrain_scan_miss_fraction must have shape "
      f"({env.num_envs},), got {tuple(value.shape)}"
    )
  if not value.is_floating_point():
    raise TypeError("terrain_scan_miss_fraction must be floating point")
  if not bool(torch.all(torch.isfinite(value)).item()):
    raise ValueError("terrain_scan_miss_fraction contains NaN or Inf")
  return value
