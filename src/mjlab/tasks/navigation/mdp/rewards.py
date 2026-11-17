"""Reward terms of the reference navigation environment.

All terms are per-step quantities, so the environment disables scaling rewards
by the step duration.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.navigation import assets
from mjlab.tasks.navigation.mdp.actions import UnicycleAction
from mjlab.tasks.navigation.mdp.commands import NavigationCommand
from mjlab.tasks.navigation.mdp.geometry import obstacle_distance
from mjlab.tasks.navigation.mdp.terminations import collision, goal_reached

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> NavigationCommand:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, NavigationCommand)
  return command


def goal_progress(
  env: ManagerBasedRlEnv, command_name: str, max_progress: float = 10.0 / 3.0
) -> torch.Tensor:
  """Reduction of the distance to the goal over the step, in meters."""
  command = _command(env, command_name)
  progress = command.prev_distance - command.goal_polar()[0]
  return progress.clamp(-max_progress, max_progress)


def goal_bearing(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Bearing of the goal relative to the heading, normalized to [0, 1]."""
  return _command(env, command_name).goal_polar()[1].abs() / math.pi


def obstacle_proximity(
  env: ManagerBasedRlEnv,
  humans_name: str | None = None,
  threshold: float = 1.0,
  robot_radius: float = assets.ROBOT_RADIUS,
) -> torch.Tensor:
  """How far the nearest obstacle intrudes into the ``threshold`` zone.

  0 when nothing is closer than ``threshold``, 1 at contact with the robot.
  """
  distance = obstacle_distance(env, humans_name=humans_name)
  return ((threshold - distance) / (threshold - robot_radius)).clamp(0.0, 1.0)


def turning_smoothness(
  env: ManagerBasedRlEnv,
  action_name: str,
  max_ang_vel: float = 1.5,
  rate_weight: float = 0.02,
  accel_weight: float = 0.01,
  max_penalty: float = 0.3,
) -> torch.Tensor:
  """Penalty on the yaw rate and on its change since the previous step."""
  action = env.action_manager.get_term(action_name)
  assert isinstance(action, UnicycleAction)
  rate = action.ang_vel / max_ang_vel
  accel = (action.ang_vel - action.prev_ang_vel) / max_ang_vel
  penalty = rate_weight * rate.square() + accel_weight * accel.square()
  return penalty.clamp(max=max_penalty)


def goal_reached_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  humans_name: str | None = None,
) -> torch.Tensor:
  """1 on the step the goal is reached without a simultaneous collision."""
  reached = goal_reached(env, command_name)
  return (reached & ~collision(env, humans_name=humans_name)).float()


def collision_penalty(
  env: ManagerBasedRlEnv, humans_name: str | None = None
) -> torch.Tensor:
  """1 on the step the robot collides."""
  return collision(env, humans_name=humans_name).float()
