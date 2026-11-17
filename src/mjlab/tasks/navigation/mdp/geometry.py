"""Exact planar clearance between the robot and everything it can hit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.tasks.navigation import assets

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_BOXES: dict[str, torch.Tensor] = {}


def _lidar_height_boxes(device: str) -> torch.Tensor:
  if device not in _BOXES:
    boxes = assets.load_hospital_map().boxes_at_height(assets.LIDAR_HEIGHT)
    _BOXES[device] = torch.tensor(boxes, dtype=torch.float, device=device)
  return _BOXES[device]


def obstacle_distance(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  humans_name: str | None = None,
) -> torch.Tensor:
  """Distance from the robot center to the nearest obstacle surface.

  This is what the smallest reading of an ideal lidar at the robot center
  would be. It is computed from joint positions rather than from the lidar
  sensor because rewards and terminations run before the sensor is refreshed
  for the new state. Shape: [B].
  """
  robot: Entity = env.scene[robot_name]
  pos = robot.data.joint_pos[:, 0:2]
  boxes = _lidar_height_boxes(env.device)
  closest = torch.maximum(torch.minimum(pos[:, None], boxes[:, 2:4]), boxes[:, 0:2])
  distance = torch.linalg.norm(pos[:, None] - closest, dim=-1).min(dim=-1).values
  if humans_name is not None:
    humans: Entity = env.scene[humans_name]
    human_pos = humans.data.joint_pos.view(env.num_envs, -1, 3)[..., 0:2]
    human_distance = torch.linalg.norm(pos[:, None] - human_pos, dim=-1)
    human_distance = (human_distance - assets.HUMAN_RADIUS).clamp(min=0.0)
    distance = torch.minimum(distance, human_distance.min(dim=-1).values)
  return distance
