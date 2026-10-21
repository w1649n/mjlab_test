"""Tests for the motor heating penalty."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp

_RATED = {r".*_Hip[XY]_joint": 6.0, r".*_Knee_joint": 9.0}
_NAMES = ("FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint")


def _make_env(torque: list[list[float]], names=_NAMES):
  robot = SimpleNamespace(
    actuator_names=names,
    data=SimpleNamespace(actuator_force=torch.tensor(torque)),
  )
  env = MagicMock()
  env.device = "cpu"
  env.scene.__getitem__.return_value = robot
  return env


def test_motor_heat_sums_load_relative_to_rated_torque():
  env = _make_env([[6.0, -12.0, 4.5], [0.0, 0.0, 0.0]])
  params = {"rated_torque": _RATED}
  cfg = RewardTermCfg(func=mdp.motor_heat_l2, weight=-1.0, params=params)
  reward = mdp.motor_heat_l2(cfg, env)
  torch.testing.assert_close(
    reward(env, rated_torque=_RATED), torch.tensor([5.25, 0.0])
  )


def test_motor_heat_requires_a_rating_for_every_actuator():
  env = _make_env([[0.0, 0.0, 0.0]], names=("FL_HipX_joint", "FL_HipY_joint", "Tail"))
  params = {"rated_torque": _RATED}
  cfg = RewardTermCfg(func=mdp.motor_heat_l2, weight=-1.0, params=params)
  with pytest.raises(ValueError, match="Tail"):
    mdp.motor_heat_l2(cfg, env)


def test_motor_heat_threshold_charges_only_load_above_rating():
  env = _make_env([[12.0, 3.0, 9.0]])
  params = {"rated_torque": _RATED, "threshold": 1.0}
  cfg = RewardTermCfg(func=mdp.motor_heat_l2, weight=-1.0, params=params)
  reward = mdp.motor_heat_l2(cfg, env)
  # Loads are 4.0, 0.25 and 1.0 times the rating; only 4.0 exceeds it.
  loads = reward(env, rated_torque=_RATED, threshold=1.0)
  torch.testing.assert_close(loads, torch.tensor([3.0]))
