from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import RayCastSensor
from mjlab.tasks.navigation.mdp.commands import NavigationCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def goal_polar(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Distance to the goal and its bearing in the robot frame. Shape: [B, 2]."""
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, NavigationCommand)
  distance, bearing = command.goal_polar()
  return torch.stack([distance, bearing], dim=-1)


def lidar_scan(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Lidar ranges, with misses reported at the sensor's maximum range.

  Shape: [B, num_rays].
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  distances = sensor.data.distances
  return torch.where(distances < 0, sensor.cfg.max_distance, distances)
