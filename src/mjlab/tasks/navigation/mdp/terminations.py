from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.navigation import assets
from mjlab.tasks.navigation.mdp.commands import NavigationCommand
from mjlab.tasks.navigation.mdp.geometry import obstacle_distance

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def goal_reached(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float = assets.ROBOT_RADIUS,
) -> torch.Tensor:
  """The robot center is within ``threshold`` of the goal."""
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, NavigationCommand)
  return command.goal_polar()[0] <= threshold


def collision(
  env: ManagerBasedRlEnv,
  humans_name: str | None = None,
  threshold: float = assets.ROBOT_RADIUS,
) -> torch.Tensor:
  """An obstacle or pedestrian is within ``threshold`` of the robot center."""
  return obstacle_distance(env, humans_name=humans_name) <= threshold
