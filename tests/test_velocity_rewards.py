"""Tests for velocity task reward functions."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

import pytest
import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastData, RayCastSensor, TerrainHeightSensor
from mjlab.tasks.velocity.mdp.rewards import (
  base_height_l2,
  feet_contact_count_standing,
  feet_excessive_air_time,
  feet_swing_height,
  standing_joint_deviation_l2,
  standing_joint_velocity_l2,
  track_angular_velocity_yaw,
  track_linear_velocity_xy,
  upright,
  vertical_velocity_l2,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz


@pytest.mark.parametrize(
  "reward", [standing_joint_deviation_l2, standing_joint_velocity_l2]
)
def test_standing_joint_costs_gate_translation_and_rotation(reward):
  data = SimpleNamespace(
    joint_pos=torch.tensor([[0.4, -0.65, 1.3]]).repeat(4, 1),
    default_joint_pos=torch.tensor([[0.0, -0.65, 1.3]]).repeat(4, 1),
    joint_vel=torch.tensor([[0.4, 0.0, 0.0]]).repeat(4, 1),
  )
  env = SimpleNamespace(
    scene={"robot": SimpleNamespace(data=data)},
    command_manager=SimpleNamespace(
      get_command=lambda _: torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.05, 0.0]]
      )
    ),
  )
  torch.testing.assert_close(reward(env, "twist"), torch.tensor([0.16, 0.0, 0.0, 0.16]))
  # Unselected joints must not contribute; resting at the nonzero default is free.
  torch.testing.assert_close(
    reward(env, "twist", asset_cfg=SceneEntityCfg("robot", joint_ids=[1, 2])),
    torch.zeros(4),
  )


def _identity_quat(B: int) -> torch.Tensor:
  """(w, x, y, z) = (1, 0, 0, 0)."""
  q = torch.zeros(B, 4)
  q[:, 0] = 1.0
  return q


def _quat_from_roll(roll_rad: float, B: int = 1) -> torch.Tensor:
  roll = torch.full((B,), roll_rad)
  zero = torch.zeros(B)
  return quat_from_euler_xyz(roll, zero, zero)


def _quat_from_pitch(pitch_rad: float, B: int = 1) -> torch.Tensor:
  pitch = torch.full((B,), pitch_rad)
  zero = torch.zeros(B)
  return quat_from_euler_xyz(zero, pitch, zero)


def _make_env_and_reward(
  terrain_sensor_names: tuple[str, ...] | None = None,
  body_quat_w: torch.Tensor | None = None,
  terrain_hit_z: float = 0.0,
  terrain_slope_x: float = 0.0,
):
  """Build mocked env + upright reward instance.

  Args:
    terrain_sensor_names: If set, enables terrain-aware mode.
    body_quat_w: [B, 4] root orientation. Defaults to identity.
    terrain_hit_z: Z value for flat terrain hits.
    terrain_slope_x: Slope in X (z = terrain_slope_x * x).
  """
  B = 1 if body_quat_w is None else body_quat_w.shape[0]
  if body_quat_w is None:
    body_quat_w = _identity_quat(B)

  # Mock asset data. Use explicit asset_cfg with no body_names so
  # body_ids stays None and the reward uses root_link_quat_w.
  asset = MagicMock()
  asset.data.root_link_quat_w = body_quat_w
  asset.data.root_link_pos_w = torch.zeros(B, 3)
  asset.data.gravity_vec_w = torch.tensor([0.0, 0.0, -1.0]).expand(B, 3)
  asset_cfg = SceneEntityCfg("robot", body_names=None, body_ids=[])

  # Mock terrain sensor if needed.
  sensors: dict = {"robot": asset}
  if terrain_sensor_names is not None:
    N = 100
    torch.manual_seed(0)
    hit_pos = torch.zeros(B, N, 3)
    hit_pos[:, :, 0] = torch.randn(B, N)
    hit_pos[:, :, 1] = torch.randn(B, N)
    hit_pos[:, :, 2] = terrain_hit_z + terrain_slope_x * hit_pos[:, :, 0]

    raycast_sensor = MagicMock(spec=RayCastSensor)
    raycast_data = RayCastData(
      distances=torch.ones(B, N),
      normals_w=torch.zeros(B, N, 3),
      hit_pos_w=hit_pos,
      pos_w=torch.zeros(B, 3),
      quat_w=torch.zeros(B, 4),
      frame_pos_w=torch.zeros(B, 1, 3),
      frame_quat_w=torch.zeros(B, 1, 4),
    )
    type(raycast_sensor).data = PropertyMock(return_value=raycast_data)
    for name in terrain_sensor_names:
      sensors[name] = raycast_sensor

  env = MagicMock()
  env.scene.__getitem__ = MagicMock(side_effect=lambda n: sensors[n])

  params: dict = {"std": 1.0, "asset_cfg": asset_cfg}
  if terrain_sensor_names is not None:
    params["terrain_sensor_names"] = terrain_sensor_names
  cfg = MagicMock(spec=RewardTermCfg)
  cfg.params = params

  reward_fn = upright(cfg, env)
  return env, reward_fn, params


def test_world_up_identity_gives_max_reward():
  """Perfectly upright robot on flat ground → reward ≈ 1."""
  env, reward, params = _make_env_and_reward()
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])
  assert r.shape == (1,)
  assert r.item() > 0.99


def test_world_up_tilted_gives_lower_reward():
  """30° roll → reward significantly below 1."""
  quat = _quat_from_roll(math.radians(30))
  env, reward, params = _make_env_and_reward(body_quat_w=quat)
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])
  assert r.item() < 0.8


def test_terrain_aware_aligned_with_slope():
  """Robot pitched to match a slope → terrain-aware reward ≈ 1."""
  slope = 0.5  # z = 0.5 * x
  tilt = math.atan(slope)  # Pitch to match slope in XZ plane.
  quat = _quat_from_pitch(-tilt)
  env, reward, params = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
    terrain_slope_x=slope,
  )
  r = reward(
    env,
    std=params["std"],
    asset_cfg=params["asset_cfg"],
    terrain_sensor_names=params["terrain_sensor_names"],
  )
  # Should be close to 1 since robot matches terrain.
  assert r.item() > 0.9


def test_terrain_aware_upright_on_slope_penalized():
  """Robot staying vertical on a slope → terrain-aware reward < 1."""
  slope = 0.5
  quat = _identity_quat(1)  # Robot is world-vertical, not matching slope.
  env, reward, params = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
    terrain_slope_x=slope,
  )
  r = reward(
    env,
    std=params["std"],
    asset_cfg=params["asset_cfg"],
    terrain_sensor_names=params["terrain_sensor_names"],
  )
  # Should be penalized since robot doesn't match terrain.
  assert r.item() < 0.95


def test_terrain_aware_flat_ground_matches_world_up():
  """On flat terrain, terrain-aware and world-up should give same reward."""
  quat = _quat_from_roll(math.radians(15))
  env_t, reward_t, params_t = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
  )
  env_w, reward_w, params_w = _make_env_and_reward(body_quat_w=quat)

  r_terrain = reward_t(
    env_t,
    std=params_t["std"],
    asset_cfg=params_t["asset_cfg"],
    terrain_sensor_names=params_t["terrain_sensor_names"],
  )
  r_world = reward_w(env_w, std=params_w["std"], asset_cfg=params_w["asset_cfg"])

  torch.testing.assert_close(r_terrain, r_world, atol=0.02, rtol=0.02)


def test_batch_consistency():
  """Multiple envs with different orientations get independent rewards."""
  B = 4
  quats = torch.zeros(B, 4)
  quats[:, 0] = 1.0  # All identity.
  # Tilt env 2 by 45°.
  quats[2] = _quat_from_roll(math.radians(45))[0]

  env, reward, params = _make_env_and_reward(body_quat_w=quats)
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])

  assert r.shape == (B,)
  # Env 0, 1, 3 should be ~1, env 2 should be lower.
  assert r[0].item() > 0.99
  assert r[1].item() > 0.99
  assert r[2].item() < 0.7
  assert r[3].item() > 0.99


# 2026-09-02 stair-training update: verify each decoupled reward uses only its axis.
def _make_velocity_env(
  linear_velocity: torch.Tensor,
  angular_velocity: torch.Tensor,
  command: torch.Tensor,
):
  asset = MagicMock()
  asset.data.root_link_lin_vel_b = linear_velocity
  asset.data.root_link_ang_vel_b = angular_velocity

  env = MagicMock()
  env.scene.__getitem__ = MagicMock(side_effect=lambda name: {"robot": asset}[name])
  env.command_manager.get_command.return_value = command
  return env


def test_track_linear_velocity_xy_ignores_vertical_velocity():
  command = torch.tensor([[1.0, -0.5, 0.7], [0.0, 0.0, -0.2]])
  linear_velocity = torch.tensor([[1.0, -0.5, 9.0], [0.3, 0.4, -8.0]])
  env = _make_velocity_env(linear_velocity, torch.zeros(2, 3), command)

  reward = track_linear_velocity_xy(env, std=0.5, command_name="twist")

  torch.testing.assert_close(reward, torch.tensor([1.0, math.exp(-1.0)]))


def test_track_angular_velocity_yaw_ignores_roll_and_pitch_rates():
  command = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
  angular_velocity = torch.tensor([[10.0, -10.0, 0.5], [-8.0, 7.0, 0.0]])
  env = _make_velocity_env(torch.zeros(2, 3), angular_velocity, command)

  reward = track_angular_velocity_yaw(env, std=0.5, command_name="twist")

  torch.testing.assert_close(reward, torch.tensor([1.0, math.exp(-1.0)]))


def test_vertical_velocity_l2_uses_only_body_z_velocity():
  linear_velocity = torch.tensor([[12.0, -6.0, 3.0], [-9.0, 4.0, -2.0]])
  env = _make_velocity_env(linear_velocity, torch.zeros(2, 3), torch.zeros(2, 3))

  torch.testing.assert_close(vertical_velocity_l2(env), torch.tensor([9.0, 4.0]))


def test_base_height_l2_uses_mean_terrain_clearance():
  height_sensor = MagicMock(spec=TerrainHeightSensor)
  height_sensor.data.heights = torch.tensor([[0.3, 0.5], [0.45, 0.55]])
  env = MagicMock()
  env.scene.__getitem__ = MagicMock(
    side_effect=lambda name: {"base_height_scan": height_sensor}[name]
  )

  cost = base_height_l2(env, target_height=0.4, height_sensor_name="base_height_scan")

  torch.testing.assert_close(cost, torch.tensor([0.0, 0.01]))


# 2026-09-02 anti-tripod update: pin continuous, capped per-foot air-time cost.
def _make_contact_reward_env(
  *,
  primary_names: tuple[str, ...],
  current_air_time: torch.Tensor | None = None,
  found: torch.Tensor | None = None,
  command: torch.Tensor | None = None,
):
  sensor = SimpleNamespace(
    primary_names=list(primary_names),
    data=SimpleNamespace(current_air_time=current_air_time, found=found),
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = command
  return SimpleNamespace(
    scene={"feet_contact": sensor},
    command_manager=command_manager,
    extras={"log": {}},
  )


def test_feet_excessive_air_time_is_continuous_and_capped_per_foot():
  env = _make_contact_reward_env(
    primary_names=("FL/foot", "FR_foot", "RL-foot", "RR foot"),
    current_air_time=torch.tensor([[0.0, 0.5, 0.75, 2.0], [0.51, 0.4, 1.0, 20.0]]),
  )

  cost = feet_excessive_air_time(
    env, sensor_name="feet_contact", max_air_time=0.5, max_excess_time=0.5
  )

  torch.testing.assert_close(cost, torch.tensor([0.3125, 0.5001]))
  log = env.extras["log"]
  torch.testing.assert_close(
    log["Metrics/anti_tripod/air_time_max"], torch.tensor(20.0)
  )
  torch.testing.assert_close(
    log["Metrics/anti_tripod/air_time_excess_max"], torch.tensor(0.5)
  )
  torch.testing.assert_close(
    log["Metrics/anti_tripod/air_time_mean/0_FL_foot"], torch.tensor(0.255)
  )
  assert "Metrics/anti_tripod/air_time_mean/3_RR_foot" in log


@pytest.mark.parametrize(
  ("max_air_time", "max_excess_time", "match"),
  [
    (-0.1, 0.5, "max_air_time"),
    (math.inf, 0.5, "max_air_time"),
    (0.5, 0.0, "max_excess_time"),
    (0.5, math.nan, "max_excess_time"),
  ],
)
def test_feet_excessive_air_time_validates_parameters(
  max_air_time: float, max_excess_time: float, match: str
):
  env = _make_contact_reward_env(
    primary_names=("foot",), current_air_time=torch.zeros(1, 1)
  )

  with pytest.raises(ValueError, match=match):
    feet_excessive_air_time(
      env,
      sensor_name="feet_contact",
      max_air_time=max_air_time,
      max_excess_time=max_excess_time,
    )


def test_feet_excessive_air_time_requires_per_primary_tracking_shape():
  missing = _make_contact_reward_env(primary_names=("FL", "FR"))
  with pytest.raises(RuntimeError, match="track_air_time=True"):
    feet_excessive_air_time(
      missing, "feet_contact", max_air_time=0.5, max_excess_time=0.5
    )

  wrong_shape = _make_contact_reward_env(
    primary_names=("FL", "FR"), current_air_time=torch.zeros(2, 3)
  )
  with pytest.raises(ValueError, match="3 columns, but 2 primaries"):
    feet_excessive_air_time(
      wrong_shape, "feet_contact", max_air_time=0.5, max_excess_time=0.5
    )


# 2026-09-02 anti-tripod update: verify stationary contact counting and slot reduction.
def test_feet_contact_count_standing_only_penalizes_stationary_commands():
  env = _make_contact_reward_env(
    primary_names=("FL", "FR", "RL", "RR"),
    found=torch.tensor(
      [
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
      ]
    ),
    command=torch.tensor(
      [
        [0.0, 0.0, 0.0],
        [0.051, 0.0, 0.0],
        [0.03, 0.0, 0.02],
      ]
    ),
  )

  cost = feet_contact_count_standing(
    env,
    sensor_name="feet_contact",
    command_name="twist",
    required_contacts=4,
    command_threshold=0.05,
  )

  torch.testing.assert_close(cost, torch.tensor([1.0, 0.0, 2.0]))
  log = env.extras["log"]
  torch.testing.assert_close(
    log["Metrics/anti_tripod/standing_missing_contacts_max"], torch.tensor(2.0)
  )
  torch.testing.assert_close(
    log["Metrics/anti_tripod/standing_contact_rate/0_FL"], torch.tensor(1.0)
  )
  torch.testing.assert_close(
    log["Metrics/anti_tripod/standing_contact_rate/3_RR"], torch.tensor(0.0)
  )


def test_feet_contact_count_standing_reduces_multiple_slots_per_foot():
  # Primary-major layout: [FL slot 0, FL slot 1, FR slot 0, FR slot 1].
  env = _make_contact_reward_env(
    primary_names=("FL", "FR"),
    found=torch.tensor([[0, 2, 0, 0], [0, 0, 1, 3]]),
    command=torch.zeros(2, 3),
  )

  cost = feet_contact_count_standing(
    env,
    sensor_name="feet_contact",
    command_name="twist",
    required_contacts=2,
  )

  torch.testing.assert_close(cost, torch.tensor([1.0, 1.0]))


@pytest.mark.parametrize("required_contacts", [0, -1, 3])
def test_feet_contact_count_standing_validates_required_contacts(
  required_contacts: int,
):
  env = _make_contact_reward_env(
    primary_names=("FL", "FR"),
    found=torch.ones(1, 2),
    command=torch.zeros(1, 3),
  )

  with pytest.raises(ValueError, match="required_contacts"):
    feet_contact_count_standing(
      env,
      sensor_name="feet_contact",
      command_name="twist",
      required_contacts=required_contacts,
    )


def test_feet_contact_count_standing_validates_sensor_and_command_shapes():
  malformed_sensor = _make_contact_reward_env(
    primary_names=("FL", "FR"),
    found=torch.ones(2, 3),
    command=torch.zeros(2, 3),
  )
  with pytest.raises(ValueError, match="cannot be grouped"):
    feet_contact_count_standing(
      malformed_sensor, "feet_contact", "twist", required_contacts=2
    )

  malformed_command = _make_contact_reward_env(
    primary_names=("FL", "FR"),
    found=torch.ones(2, 2),
    command=torch.zeros(1, 3),
  )
  with pytest.raises(ValueError, match="matching the contact batch"):
    feet_contact_count_standing(
      malformed_command, "feet_contact", "twist", required_contacts=2
    )


# 2026-09-02 anti-tripod update: swing peaks must not leak across episode resets.
def test_feet_swing_height_partial_reset_only_clears_selected_envs():
  height_sensor = MagicMock(spec=TerrainHeightSensor)
  height_sensor.num_frames = 4
  env = MagicMock()
  env.scene.__getitem__ = MagicMock(
    side_effect=lambda name: {"foot_height_scan": height_sensor}[name]
  )
  env.num_envs = 4
  env.device = "cpu"
  env.step_dt = 0.02
  cfg = MagicMock(spec=RewardTermCfg)
  cfg.params = {"height_sensor_name": "foot_height_scan"}
  reward = feet_swing_height(cfg, env)
  reward.peak_heights[:] = torch.tensor(
    [
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
      [0.9, 1.0, 1.1, 1.2],
      [1.3, 1.4, 1.5, 1.6],
    ]
  )

  reward.reset(torch.tensor([1, 3]))

  torch.testing.assert_close(
    reward.peak_heights,
    torch.tensor(
      [
        [0.1, 0.2, 0.3, 0.4],
        [0.0, 0.0, 0.0, 0.0],
        [0.9, 1.0, 1.1, 1.2],
        [0.0, 0.0, 0.0, 0.0],
      ]
    ),
  )
