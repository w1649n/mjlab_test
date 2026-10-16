"""Tests for the G23 stair heading-hold reward and stair-aligned reset."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import events as velocity_events

_STAIRS = ("pyramid_stairs",)


def _make_env(terrain_types: list[int], curriculum: bool = True):
  num_envs = len(terrain_types)
  generator = SimpleNamespace(
    sub_terrains={"flat": None, "pyramid_stairs": None}, curriculum=curriculum
  )
  terrain = SimpleNamespace(
    terrain_origins=torch.zeros(1, 2, 3),
    terrain_types=torch.tensor(terrain_types),
    cfg=SimpleNamespace(terrain_generator=generator),
  )
  robot = SimpleNamespace(data=SimpleNamespace(heading_w=torch.zeros(num_envs)))
  command = SimpleNamespace(
    command_counter=torch.ones(num_envs, dtype=torch.long),
    is_forward_env=torch.ones(num_envs, dtype=torch.bool),
    is_backward_env=torch.zeros(num_envs, dtype=torch.bool),
  )
  env = MagicMock()
  env.num_envs = num_envs
  env.device = "cpu"
  env.scene.terrain = terrain
  env.scene.__getitem__.return_value = robot
  env.command_manager.get_term.return_value = command
  return env, robot, command


def _make_heading_reward(env):
  params = {"command_name": "twist", "terrain_names": _STAIRS}
  cfg = RewardTermCfg(
    func=mdp.straight_heading_deviation_l2, weight=-1.0, params=params
  )
  return mdp.straight_heading_deviation_l2(cfg, env), params


def test_heading_deviation_charges_drift_within_a_straight_stair_segment():
  # Env 0 is on flat ground, envs 1-3 are on stairs; env 3 is not straight.
  env, robot, command = _make_env([0, 1, 1, 1])
  command.is_forward_env[3] = False
  reward, params = _make_heading_reward(env)

  robot.data.heading_w = torch.tensor([0.3, 0.3, 3.1, 0.3])
  assert torch.equal(reward(env, **params), torch.zeros(4))

  robot.data.heading_w = torch.tensor([0.5, 0.5, -3.1, 0.5])
  expected = torch.tensor([0.0, 0.2**2, (2 * math.pi - 6.2) ** 2, 0.0])
  torch.testing.assert_close(reward(env, **params), expected)

  # A resampled command or an env reset starts a new reference heading.
  command.command_counter[1] += 1
  reward.reset(torch.tensor([2]))
  robot.data.heading_w = torch.tensor([0.9, 0.9, 1.0, 0.9])
  assert torch.equal(reward(env, **params), torch.zeros(4))


def test_heading_deviation_applies_everywhere_without_terrain_types():
  env, robot, _ = _make_env([0, 1], curriculum=False)
  reward, params = _make_heading_reward(env)
  reward(env, **params)
  robot.data.heading_w = torch.tensor([0.1, -0.1])
  torch.testing.assert_close(reward(env, **params), torch.full((2,), 0.01))


def test_stair_aligned_reset_faces_a_stair_axis(monkeypatch):
  calls = []
  monkeypatch.setattr(
    velocity_events,
    "reset_root_state_uniform",
    lambda env, ids, ranges, *_: calls.append((ids.tolist(), ranges["yaw"])),
  )
  env, _, _ = _make_env([0, 1] * 50)
  mdp.reset_root_state_stair_aligned(
    env, None, {"yaw": (-3.14, 3.14)}, _STAIRS, yaw_noise=0.1
  )

  reset_ids = sorted(i for ids, _ in calls for i in ids)
  assert reset_ids == list(range(100))
  for ids, (yaw_min, yaw_max) in calls:
    if ids[0] % 2 == 0:
      assert (yaw_min, yaw_max) == (-3.14, 3.14)
      continue
    assert all(i % 2 == 1 for i in ids)
    center = 0.5 * (yaw_min + yaw_max)
    assert yaw_max - yaw_min == pytest.approx(0.2)
    assert center / (math.pi / 2) == pytest.approx(round(center / (math.pi / 2)))
