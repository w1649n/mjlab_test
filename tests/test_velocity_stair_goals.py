"""Tests for the G23 stair-goals command, curriculum, and reward changes."""

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
import torch
from conftest import get_test_device, load_fixture_xml, make_scene_and_sim

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_step_height_lookup
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _make_command(device: str, num_envs: int = 2, **cfg_kwargs):
  scene, sim = make_scene_and_sim(
    device, load_fixture_xml("floating_base_articulated"), sensors=(), num_envs=num_envs
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      scene=scene, sim=sim, num_envs=num_envs, device=device, step_dt=0.02
    ),
  )
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(100.0, 100.0),
    rel_standing_envs=0.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.5, 0.5),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.5, 0.5),
      heading=(-math.pi, math.pi) if cfg_kwargs.get("heading_command") else None,
    ),
    **cfg_kwargs,
  )
  term = cfg.build(env)
  sim.forward()
  term.reset(torch.arange(num_envs, device=device))
  return term


def _set_yaw(term, yaw: float) -> None:
  num_envs = term.num_envs
  pose = term.robot.data.root_link_pose_w.clone()
  pose[:, 3:7] = torch.tensor(
    [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=term.device
  )
  term.robot.write_root_link_pose_to_sim(pose)
  term._env.sim.forward()
  assert term.robot.data.heading_w.shape == (num_envs,)


def test_directional_heading_hold_steers_back_to_the_starting_heading(device):
  term = _make_command(
    device,
    heading_command=True,
    rel_heading_envs=0.0,
    rel_forward_envs=1.0,
    directional_heading_hold=True,
    heading_control_stiffness=2.0,
  )
  # The reset-path resample must not read the stale heading; the next command
  # update latches the current one.
  _set_yaw(term, 0.7)
  term.compute(dt=0.0)
  assert term.is_forward_env.all() and term.is_heading_env.all()
  torch.testing.assert_close(term.heading_target, term.robot.data.heading_w)
  assert torch.allclose(term.command[:, 2], torch.zeros(2, device=device))

  # Drifting 0.1 rad left asks for a 0.2 rad/s right turn; the latch holds.
  _set_yaw(term, 0.8)
  term.compute(dt=0.0)
  expected = torch.full((2,), -0.2, device=device)
  torch.testing.assert_close(term.command[:, 2], expected, atol=1e-5, rtol=0.0)
  assert torch.allclose(term.command[:, 0], torch.full((2,), 0.5, device=device))


def test_directional_heading_hold_requires_heading_commands():
  with pytest.raises(ValueError, match="directional_heading_hold"):
    UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0, 1.0),
      directional_heading_hold=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
      ),
    )


def test_command_progress_and_stationary_time(device):
  term = _make_command(device)
  term.vel_command_b[:] = torch.tensor(
    [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device
  )
  velocity = torch.zeros(2, 6, device=device)
  velocity[0, 0] = 0.8  # Faster than commanded: progress is capped.
  term.robot.write_root_link_velocity_to_sim(velocity)
  term._env.sim.forward()

  term.compute(torch.tensor([0.1, 0.1], device=device))
  term.compute(torch.tensor([0.1, 0.1], device=device))
  path, progress = term.get_command_progress(torch.arange(2, device=device))
  torch.testing.assert_close(path, torch.tensor([0.1, 0.0], device=device))
  torch.testing.assert_close(progress, torch.tensor([0.1, 0.0], device=device))
  # Only the zero command accumulates stationary time.
  torch.testing.assert_close(
    term.stationary_time, torch.tensor([0.0, 0.2], device=device)
  )

  term.reset(torch.tensor([1], device=device))
  assert term.stationary_time[1].item() == 0.0
  assert term.get_command_progress(torch.arange(2, device=device))[0][0] > 0.0


class _FakeTerrain:
  def __init__(self, num_envs: int):
    self.terrain_levels = torch.full((num_envs,), 3, dtype=torch.long)
    self.terrain_types = torch.zeros(num_envs, dtype=torch.long)
    self.max_terrain_level = 10
    self.terrain_origins = torch.zeros(10, 1, 3)
    self.env_origins = torch.zeros(num_envs, 3)
    self.cfg = SimpleNamespace(
      terrain_generator=SimpleNamespace(size=(8.0, 8.0), sub_terrains={})
    )

  def update_env_origins(self, env_ids, move_up, move_down):
    self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down


def test_terrain_levels_tracked_moves_on_command_following():
  # 0: followed 90% of 3 m; 1: followed 30%; 2: commanded too little to judge;
  # 3: followed well but was terminated; 4: left the patch.
  terrain = _FakeTerrain(num_envs=5)
  command = SimpleNamespace(
    get_command_progress=lambda ids: (
      torch.tensor([3.0, 3.0, 1.0, 3.0, 0.0])[ids],
      torch.tensor([2.7, 0.9, 0.0, 2.7, 0.0])[ids],
    )
  )
  asset = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[1.0, 0.0, 0.0]] * 4 + [[5.0, 0.0, 0.0]])
    )
  )
  env = MagicMock()
  env.common_step_counter = 1
  env.scene.__getitem__.return_value = asset
  env.scene.terrain = terrain
  env.scene.env_origins = torch.zeros(5, 3)
  env.command_manager.get_term.return_value = command
  env.termination_manager.terminated = torch.tensor([False, False, False, True, False])

  mdp.terrain_levels_tracked(env, torch.arange(5), command_name="twist")
  assert terrain.terrain_levels.tolist() == [4, 2, 3, 2, 4]


def _standing_env(stationary_time: list[float], command: list[list[float]]):
  robot = SimpleNamespace(
    data=SimpleNamespace(
      joint_vel=torch.ones(len(command), 2),
      joint_pos=torch.ones(len(command), 2),
      default_joint_pos=torch.zeros(len(command), 2),
    )
  )
  env = MagicMock()
  env.scene.__getitem__.return_value = robot
  env.command_manager.get_command.return_value = torch.tensor(command)
  env.command_manager.get_term.return_value = SimpleNamespace(
    stationary_time=torch.tensor(stationary_time)
  )
  return env


def test_standing_penalties_wait_for_the_settle_time():
  env = _standing_env([0.5, 1.5, 3.0], [[0.0, 0.0, 0.0]] * 2 + [[0.5, 0.0, 0.0]])
  velocity = mdp.standing_joint_velocity_l2(
    env, "twist", settle_time_s=1.0, asset_cfg=SceneEntityCfg("robot")
  )
  torch.testing.assert_close(velocity, torch.tensor([0.0, 2.0, 0.0]))
  # Without a settle time the stationary envs are charged at once.
  deviation = mdp.standing_joint_deviation_l2(
    env, "twist", asset_cfg=SceneEntityCfg("robot")
  )
  torch.testing.assert_close(deviation, torch.tensor([2.0, 2.0, 0.0]))


def test_standing_deviation_applies_only_on_level_ground():
  env = _standing_env([2.0, 2.0], [[0.0, 0.0, 0.0]] * 2)
  robot = env.scene.__getitem__.return_value
  sensor = MagicMock(spec=TerrainHeightSensor)
  # Feet 3 cm above the ground; env 1 stands across a 20 cm riser.
  foot_z = torch.tensor([[0.03] * 4, [0.23, 0.23, 0.03, 0.03]])
  sensor.data = SimpleNamespace(
    frame_pos_w=torch.stack([torch.zeros(2, 4), torch.zeros(2, 4), foot_z], dim=-1),
    heights=torch.full((2, 4), 0.03),
  )
  env.scene.__getitem__.side_effect = {"robot": robot, "feet": sensor}.__getitem__
  deviation = mdp.standing_joint_deviation_l2(
    env, "twist", foot_height_sensor_name="feet", asset_cfg=SceneEntityCfg("robot")
  )
  torch.testing.assert_close(deviation, torch.tensor([2.0, 0.0]))


def test_settle_time_needs_a_stationary_timer():
  env = _standing_env([0.0], [[0.0, 0.0, 0.0]])
  env.command_manager.get_term.return_value = SimpleNamespace()
  with pytest.raises(TypeError, match="stationary_time"):
    mdp.standing_joint_velocity_l2(
      env, "twist", settle_time_s=1.0, asset_cfg=SceneEntityCfg("robot")
    )


def _stair_env(levels: list[int], types: list[int], curriculum: bool = True):
  stairs = SimpleNamespace(step_height_range=(0.05, 0.25))
  generator = SimpleNamespace(
    sub_terrains={"flat": SimpleNamespace(), "pyramid_stairs": stairs},
    curriculum=curriculum,
    num_rows=5,
    difficulty_range=(0.0, 1.0),
  )
  terrain = SimpleNamespace(
    terrain_origins=torch.zeros(5, 2, 3),
    terrain_levels=torch.tensor(levels),
    terrain_types=torch.tensor(types),
    cfg=SimpleNamespace(terrain_generator=generator),
  )
  env = MagicMock()
  env.device = "cpu"
  env.scene.terrain = terrain
  return env


def test_stair_clearance_target_follows_the_riser_height():
  env = _stair_env(levels=[4, 0, 2, 4], types=[0, 1, 1, 1])
  lookup = terrain_step_height_lookup(env, ("pyramid_stairs",))
  assert lookup is not None
  torch.testing.assert_close(lookup[:, 1], torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25]))
  assert torch.all(lookup[:, 0] == 0.0)

  params = {
    "target_height": 0.16,
    "terrain_names": ("pyramid_stairs",),
    "step_height_margin": 0.04,
    "max_target_height": 0.24,
  }
  cfg = RewardTermCfg(
    func=mdp.feet_clearance_stair_adaptive, weight=-1.0, params=params
  )
  reward = mdp.feet_clearance_stair_adaptive(cfg, env)
  target = reward._target.target(env, 0.16, 0.04, 0.24)
  assert isinstance(target, torch.Tensor)
  # Flat keeps the base target; 5 cm and 15 cm risers sit at or near it; 25 cm caps.
  torch.testing.assert_close(target[:, 0], torch.tensor([0.16, 0.16, 0.19, 0.24]))

  random_grid = _stair_env(levels=[0], types=[1], curriculum=False)
  assert terrain_step_height_lookup(random_grid, ("pyramid_stairs",)) is None
