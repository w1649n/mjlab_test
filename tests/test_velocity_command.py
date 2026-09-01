"""Tests for the velocity command's initial-velocity injection.

The init_velocity_prob path runs inside the reset pipeline, after reset
events wrote the new pose to qpos but before sim.forward(), so it must not
read (or write back) derived kinematics.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from conftest import get_test_device, load_fixture_xml, make_scene_and_sim

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _make_velocity_command(
  device: str,
  *,
  num_envs: int = 3,
  resampling_time_range: tuple[float, float] = (100.0, 100.0),
  ranges: UniformVelocityCommandCfg.Ranges | None = None,
  heading_command: bool = False,
):
  scene, sim = make_scene_and_sim(
    device,
    load_fixture_xml("floating_base_articulated"),
    sensors=(),
    num_envs=num_envs,
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      scene=scene, sim=sim, num_envs=num_envs, device=device, step_dt=0.02
    ),
  )
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=resampling_time_range,
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    rel_world_envs=0.0,
    rel_forward_envs=0.0,
    heading_command=heading_command,
    ranges=ranges
    or UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.4, 0.8),
      lin_vel_y=(-0.2, 0.3),
      ang_vel_z=(-0.5, 0.6),
      heading=(-1.0, 1.0) if heading_command else None,
    ),
  )
  term = cfg.build(env)
  sim.forward()
  term.reset(torch.arange(num_envs, device=device))
  return term


def test_init_velocity_preserves_fresh_reset_pose(device):
  scene, sim = make_scene_and_sim(
    device, load_fixture_xml("floating_base_articulated"), sensors=(), num_envs=2
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(scene=scene, sim=sim, num_envs=2, device=device),
  )
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1e9, 1e9),
    init_velocity_prob=1.0,
    rel_heading_envs=0.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.5, 0.5), lin_vel_y=(0.2, 0.2), ang_vel_z=(0.0, 0.0)
    ),
  )
  term = cfg.build(env)
  robot = scene["robot"]
  env_ids = torch.arange(2, device=device)

  # Derived kinematics now hold the spawn pose (the "previous episode" state).
  sim.forward()

  # Emulate a reset event: write a fresh pose to qpos, no forward yet.
  pose = torch.tensor(
    [
      [1.0, 2.0, 1.5, 1.0, 0.0, 0.0, 0.0],
      [3.0, -1.0, 1.5, 1.0, 0.0, 0.0, 0.0],
    ],
    device=device,
  )
  robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)

  term.reset(env_ids=env_ids)
  sim.forward()

  # The fresh reset pose survives. Before the fix, the init-velocity path
  # wrote the stale pre-reset pose back into the sim.
  assert torch.allclose(robot.data.root_link_pos_w, pose[:, :3], atol=1e-6)

  # Planar velocity matches the sampled command (identity orientation, so
  # body frame equals world frame).
  assert torch.allclose(
    robot.data.root_link_lin_vel_b[:, :2],
    term.vel_command_b[:, :2],
    atol=1e-5,
  )


def test_mid_episode_resample_does_not_write_velocity(device):
  """Init velocity applies on reset only; a timer-expiry resample runs after
  step()'s forward and must not write sim state."""
  scene, sim = make_scene_and_sim(
    device, load_fixture_xml("floating_base_articulated"), sensors=(), num_envs=2
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(scene=scene, sim=sim, num_envs=2, device=device, step_dt=0.02),
  )
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(0.001, 0.001),  # Expires on the first compute.
    init_velocity_prob=1.0,
    rel_heading_envs=0.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.7, 0.7), lin_vel_y=(0.3, 0.3), ang_vel_z=(0.0, 0.0)
    ),
  )
  term = cfg.build(env)
  robot = scene["robot"]
  env_ids = torch.arange(2, device=device)

  term.reset(env_ids=env_ids)
  sim.forward()

  # Mid-episode the robot has decelerated to rest.
  robot.write_root_link_velocity_to_sim(
    torch.zeros(2, 6, device=device), env_ids=env_ids
  )
  sim.forward()

  # Step's command compute: the 1 ms timer expires and resamples.
  counter = term.command_counter.clone()
  term.compute(dt=1.0)
  assert (term.command_counter > counter).all()

  # Only the command changed; sim velocity is untouched.
  qvel_lin = sim.data.qvel[:, robot.indexing.free_joint_v_adr[:3]]
  assert torch.allclose(qvel_lin, torch.zeros_like(qvel_lin), atol=1e-6)
  assert torch.allclose(
    robot.data.root_link_lin_vel_b,
    torch.zeros_like(robot.data.root_link_lin_vel_b),
    atol=1e-6,
  )


def test_manual_command_applies_immediately_and_clamps(device):
  term = _make_velocity_command(device, num_envs=1)
  result = term.set_manual_command(0, (2.0, -1.0, 0.9))

  assert result == pytest.approx((0.8, -0.2, 0.6))
  assert term.command[0].tolist() == pytest.approx((0.8, -0.2, 0.6))

  assert term.set_manual_command(0, (-2.0, 1.0, -0.9)) == pytest.approx(
    (-0.4, 0.3, -0.5)
  )
  assert term.command[0].tolist() == pytest.approx((-0.4, 0.3, -0.5))


def test_manual_command_survives_timer_resample_and_reset(device):
  term = _make_velocity_command(
    device, num_envs=2, resampling_time_range=(0.001, 0.001)
  )
  command = (0.35, -0.15, 0.45)
  term.set_manual_command(0, command)

  counter = term.command_counter.clone()
  term.compute(dt=1.0)
  assert (term.command_counter > counter).all()
  assert term.command[0].tolist() == pytest.approx(command)

  env_ids = torch.tensor([0], device=device)
  term.reset(env_ids)
  term.compute(dt=0.0, env_ids=env_ids)
  assert term.command[0].tolist() == pytest.approx(command)


def test_manual_command_only_affects_requested_environment(device):
  term = _make_velocity_command(device, num_envs=3)
  baseline = torch.tensor(
    [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
    device=device,
  )
  term.vel_command_b[:] = baseline

  term.set_manual_command(1, (-0.25, -0.1, -0.35))
  term.compute(dt=0.0)

  assert torch.equal(term.command[0], baseline[0])
  assert term.command[1].tolist() == pytest.approx((-0.25, -0.1, -0.35))
  assert torch.equal(term.command[2], baseline[2])


def test_releasing_manual_command_immediately_resamples(device, monkeypatch):
  term = _make_velocity_command(device, num_envs=2)
  term.set_manual_command(0, (0.25, 0.15, -0.25))

  fresh = torch.tensor((0.7, -0.1, 0.4), device=device)
  resampled_envs: list[list[int]] = []

  def deterministic_resample(env_ids: torch.Tensor) -> None:
    resampled_envs.append(env_ids.tolist())
    term.vel_command_b[env_ids] = fresh

  monkeypatch.setattr(term, "_resample_command", deterministic_resample)
  counter = term.command_counter.clone()

  assert term.set_manual_command(0, None) is None

  assert resampled_envs == [[0]]
  assert term.command_counter[0] == counter[0] + 1
  assert term.command_counter[1] == counter[1]
  assert term.command[0].tolist() == pytest.approx(fresh.tolist())


def test_manual_command_overrides_standing_heading_and_world_updates(device):
  term = _make_velocity_command(device, num_envs=1, heading_command=True)
  term.is_standing_env[0] = True
  term.is_heading_env[0] = True
  term.is_world_env[0] = True
  term.heading_target[0] = 0.9
  term.vel_command_w[0] = torch.tensor((0.6, 0.2, 0.0), device=device)

  command = (-0.3, 0.25, -0.45)
  term.set_manual_command(0, command)

  assert term.command[0].tolist() == pytest.approx(command)

  # Even if another command subtype marks the env again later, the persistent
  # manual override must still be applied last on every compute.
  term.is_standing_env[0] = True
  term.is_heading_env[0] = True
  term.is_world_env[0] = True
  term.compute(dt=0.0)
  assert term.command[0].tolist() == pytest.approx(command)


def test_manual_command_disables_viser_joystick_override(device):
  term = _make_velocity_command(device, num_envs=1)
  joystick = SimpleNamespace(value=True)
  term._joystick_enabled = cast(Any, joystick)

  term.set_manual_command(0, (0.2, 0.1, -0.1))

  assert joystick.value is False


def test_manual_command_invalidates_observation_cache(device):
  term = _make_velocity_command(device, num_envs=1)
  observation_manager = SimpleNamespace(invalidate_cache=MagicMock())
  term._env.observation_manager = cast(Any, observation_manager)

  term.set_manual_command(0, (0.2, 0.1, -0.1))
  observation_manager.invalidate_cache.assert_called_once_with()

  observation_manager.invalidate_cache.reset_mock()
  term.set_manual_command(0, None)
  observation_manager.invalidate_cache.assert_called_once_with()


@pytest.mark.parametrize("env_idx", [-1, 2])
def test_manual_command_rejects_bad_environment_index(device, env_idx):
  term = _make_velocity_command(device, num_envs=2)

  with pytest.raises(IndexError):
    term.set_manual_command(env_idx, (0.0, 0.0, 0.0))


@pytest.mark.parametrize("command", [(0.0, 0.0), (0.0, 0.0, 0.0, 0.0)])
def test_manual_command_rejects_bad_command_length(device, command):
  term = _make_velocity_command(device, num_envs=1)

  with pytest.raises(ValueError):
    term.set_manual_command(0, command)


@pytest.mark.parametrize(
  "command",
  [
    (float("nan"), 0.0, 0.0),
    (0.0, float("inf"), 0.0),
    (0.0, 0.0, float("-inf")),
  ],
)
def test_manual_command_rejects_nonfinite_command(device, command):
  term = _make_velocity_command(device, num_envs=1)

  with pytest.raises(ValueError):
    term.set_manual_command(0, command)
