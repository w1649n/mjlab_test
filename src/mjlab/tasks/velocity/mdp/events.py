"""Velocity-task reset events."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import reset_root_state_uniform, resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_type_lookup

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_root_state_stair_aligned(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  terrain_names: tuple[str, ...],
  yaw_noise: float,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset like ``reset_root_state_uniform``, facing straight along stair axes.

  Pyramid stair edges are axis-aligned, so on ``terrain_names`` the sampled yaw is
  replaced by a random multiple of 90 degrees plus ``U(-yaw_noise, yaw_noise)``. A
  straight command then climbs or descends the flight head-on. Envs on other
  terrains keep ``pose_range`` unchanged.
  """
  env_ids = resolve_env_ids(env, env_ids)
  lookup = terrain_type_lookup(env, terrain_names)
  # -1 keeps the configured yaw range; 0..3 selects the stair axis to face.
  quadrant = torch.randint(0, 4, (len(env_ids),), device=env.device)
  if lookup is not None:
    terrain = env.scene.terrain
    assert terrain is not None
    quadrant[~lookup[terrain.terrain_types[env_ids]]] = -1

  for k in range(-1, 4):
    ids = env_ids[quadrant == k]
    if len(ids) == 0:
      continue
    if k < 0:
      ranges = pose_range
    else:
      yaw = k * math.pi / 2
      ranges = {**pose_range, "yaw": (yaw - yaw_noise, yaw + yaw_noise)}
    reset_root_state_uniform(env, ids, ranges, velocity_range, asset_cfg)
