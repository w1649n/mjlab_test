"""Tests for the velocity-command curriculum."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.curriculums import (
  PerformanceGatedVelocityCurriculum,
  PerformanceVelocityStage,
  VelocityStage,
  commands_vel,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

_BACKWARD_STAGE_2_STEP = 12_450 * 24
_BACKWARD_STAGE_3_STEP = 12_950 * 24
_VELOCITY_STAGES: list[VelocityStage] = [
  {
    "step": 0,
    "lin_vel_x": (-1.0, 1.0),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.5, 0.5),
    "rel_backward_envs": 0.1,
  },
  {
    "step": 8000 * 24,
    "lin_vel_x": (-1.25, 1.25),
    "lin_vel_y": (-0.4, 0.4),
    "ang_vel_z": (-0.7, 0.7),
  },
  {"step": _BACKWARD_STAGE_2_STEP, "rel_backward_envs": 0.2},
  {"step": _BACKWARD_STAGE_3_STEP, "rel_backward_envs": 0.3},
]


def _make_env(common_step_counter: int):
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 1.0),
    rel_forward_envs=0.4,
    rel_backward_envs=0.1,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.5, 1.5),
      lin_vel_y=(-0.2, 0.2),
      ang_vel_z=(-0.5, 0.5),
    ),
  )
  command_term = SimpleNamespace(cfg=cfg)
  command_manager = SimpleNamespace(get_term=lambda _name: command_term)
  env = cast(
    Any,
    SimpleNamespace(
      common_step_counter=common_step_counter,
      command_manager=command_manager,
    ),
  )
  return env, cfg


@pytest.mark.parametrize(
  ("common_step_counter", "expected_backward_fraction"),
  [
    (_BACKWARD_STAGE_2_STEP - 1, 0.1),
    (_BACKWARD_STAGE_2_STEP, 0.2),
    (_BACKWARD_STAGE_3_STEP - 1, 0.2),
    (_BACKWARD_STAGE_3_STEP, 0.3),
  ],
)
def test_backward_fraction_advances_at_configured_steps(
  common_step_counter, expected_backward_fraction
):
  env, cfg = _make_env(common_step_counter)

  state = commands_vel(
    env,
    torch.empty(0, dtype=torch.long),
    command_name="twist",
    velocity_stages=_VELOCITY_STAGES,
  )

  assert cfg.rel_forward_envs == pytest.approx(0.4)
  assert cfg.rel_backward_envs == pytest.approx(expected_backward_fraction)
  assert state["rel_backward_envs"].item() == pytest.approx(expected_backward_fraction)
  # Backward-only milestones must not overwrite the active velocity stage.
  assert cfg.ranges.lin_vel_x == (-1.25, 1.25)
  assert cfg.ranges.lin_vel_y == (-0.4, 0.4)
  assert cfg.ranges.ang_vel_z == (-0.7, 0.7)


def test_backward_curriculum_rejects_invalid_directional_fraction():
  env, cfg = _make_env(_BACKWARD_STAGE_2_STEP)
  invalid_stages: list[VelocityStage] = [
    {"step": 0, "rel_backward_envs": 0.1},
    {"step": _BACKWARD_STAGE_2_STEP, "rel_backward_envs": 0.7},
  ]

  with pytest.raises(ValueError, match="must not exceed 1"):
    commands_vel(
      env,
      torch.empty(0, dtype=torch.long),
      command_name="twist",
      velocity_stages=invalid_stages,
    )

  assert cfg.rel_backward_envs == pytest.approx(0.1)


# 2026-09-02 stair-training update: exercise the stateful, performance-gated
# replacement without constructing a MuJoCo environment.
_STAIR_TERRAINS = ("pyramid_stairs", "pyramid_stairs_inv")
_PERFORMANCE_STAGES: list[PerformanceVelocityStage] = [
  {
    "min_terrain_level": 0,
    "lin_vel_x": (-1.0, 1.0),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.5, 0.5),
    "rel_forward_envs": 0.4,
    "rel_backward_envs": 0.1,
  },
  {
    "min_terrain_level": 3,
    "lin_vel_x": (-1.1, 1.1),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.6, 0.6),
    "rel_forward_envs": 0.4,
    "rel_backward_envs": 0.2,
  },
  {
    "min_terrain_level": 5,
    "lin_vel_x": (-1.25, 1.25),
    "lin_vel_y": (-0.4, 0.4),
    "ang_vel_z": (-0.7, 0.7),
    "rel_forward_envs": 0.4,
    "rel_backward_envs": 0.3,
  },
]


class _FakeDirectionalCommand:
  def __init__(self, cfg: UniformVelocityCommandCfg, num_envs: int):
    self.cfg = cfg
    self.forward_sum = torch.full((num_envs,), 9.0)
    self.forward_count = torch.full((num_envs,), 10, dtype=torch.long)
    self.backward_sum = torch.full((num_envs,), 9.0)
    self.backward_count = torch.full((num_envs,), 10, dtype=torch.long)
    self.restore_resample_count = 0

  def get_directional_tracking_stats(
    self, env_ids: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    return {
      "forward_tracking_sum": self.forward_sum[env_ids],
      "forward_tracking_count": self.forward_count[env_ids],
      "backward_tracking_sum": self.backward_sum[env_ids],
      "backward_tracking_count": self.backward_count[env_ids],
    }

  def resample_after_curriculum_restore(self) -> None:
    self.restore_resample_count += 1


def _make_performance_gate(
  *,
  common_step_counter: int = 0,
  warmup_steps: int = 10,
  min_stage_steps: int = 5,
  required_passes: int = 2,
  min_episodes_per_terrain: int = 2,
  min_directional_samples_per_terrain: int = 20,
):
  num_envs = 4
  command_cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 1.0),
    rel_forward_envs=0.2,
    rel_backward_envs=0.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-2.0, 2.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
    ),
  )
  command_term = _FakeDirectionalCommand(command_cfg, num_envs)
  command_manager = SimpleNamespace(get_term=lambda _name: command_term)

  terrain_origins = torch.zeros(10, len(_STAIR_TERRAINS), 3)
  env_origins = torch.zeros(num_envs, 3)
  terrain_generator = SimpleNamespace(
    size=(8.0, 8.0),
    sub_terrains={name: object() for name in _STAIR_TERRAINS},
  )
  terrain = SimpleNamespace(
    terrain_origins=terrain_origins,
    terrain_levels=torch.full((num_envs,), 9, dtype=torch.long),
    terrain_types=torch.tensor([0, 0, 1, 1]),
    env_origins=env_origins,
    cfg=SimpleNamespace(terrain_generator=terrain_generator),
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[5.0, 0.0, 0.0]] * num_envs, dtype=torch.float)
    )
  )
  foot_contact_sensor = MagicMock(spec=ContactSensor)
  foot_contact_sensor.data.current_air_time = torch.zeros(num_envs, 4)

  class _Scene:
    def __init__(self):
      self.terrain = terrain

    def __getitem__(self, name):
      return {
        "robot": robot,
        "feet_ground_contact": foot_contact_sensor,
      }[name]

  time_out = torch.ones(num_envs, dtype=torch.bool)
  illegal_contact = torch.zeros(num_envs, dtype=torch.bool)
  out_of_terrain_bounds = torch.zeros(num_envs, dtype=torch.bool)
  termination_terms = {
    "time_out": time_out,
    "illegal_contact": illegal_contact,
    "out_of_terrain_bounds": out_of_terrain_bounds,
  }
  termination_manager = SimpleNamespace(
    active_terms=list(termination_terms),
    get_term=termination_terms.__getitem__,
    get_term_cfg=lambda name: SimpleNamespace(time_out=name != "illegal_contact"),
  )
  env = cast(
    Any,
    SimpleNamespace(
      common_step_counter=common_step_counter,
      command_manager=command_manager,
      termination_manager=termination_manager,
      scene=_Scene(),
      episode_length_buf=torch.ones(num_envs, dtype=torch.long),
      reset_buf=torch.ones(num_envs, dtype=torch.bool),
    ),
  )
  term_cfg = CurriculumTermCfg(
    func=PerformanceGatedVelocityCurriculum,
    params={
      "command_name": "twist",
      "stages": _PERFORMANCE_STAGES,
      "terrain_names": _STAIR_TERRAINS,
      "min_episodes_per_terrain": min_episodes_per_terrain,
      "min_directional_samples_per_terrain": (min_directional_samples_per_terrain),
      "min_safe_success_rate": 0.8,
      "min_directional_tracking_score": 0.8,
      "warmup_steps": warmup_steps,
      "min_stage_steps": min_stage_steps,
      "required_passes": required_passes,
      "success_termination_name": "time_out",
      "contact_sensor_name": "feet_ground_contact",
      "max_foot_air_time": 0.8,
    },
  )
  gate = PerformanceGatedVelocityCurriculum(term_cfg, env)
  return gate, env, command_term, command_cfg, time_out


def _assert_stage_applied(
  cfg: UniformVelocityCommandCfg, stage: PerformanceVelocityStage
) -> None:
  assert cfg.ranges.lin_vel_x == stage["lin_vel_x"]
  assert cfg.ranges.lin_vel_y == stage["lin_vel_y"]
  assert cfg.ranges.ang_vel_z == stage["ang_vel_z"]
  assert cfg.rel_forward_envs == pytest.approx(stage["rel_forward_envs"])
  assert cfg.rel_backward_envs == pytest.approx(stage["rel_backward_envs"])


def test_performance_gate_requires_warmup_and_two_independent_windows():
  gate, env, _command, command_cfg, _time_out = _make_performance_gate()
  env_ids = torch.arange(4)

  # Construction applies a complete stage-0 snapshot.
  _assert_stage_applied(command_cfg, _PERFORMANCE_STAGES[0])
  env.common_step_counter = 9
  gate(env, env_ids)
  assert gate.stage_index == 0
  assert gate.state_dict()["window"][_STAIR_TERRAINS[0]]["episode_count"] == 0

  env.common_step_counter = 10
  first = gate(env, env_ids)
  assert gate.stage_index == 0
  assert first["pass_streak"].item() == 1
  assert first[f"last/{_STAIR_TERRAINS[0]}/safe_success_rate"].item() == 1.0
  assert first[f"last/{_STAIR_TERRAINS[1]}/forward_tracking_score"].item() == (
    pytest.approx(0.9)
  )

  env.common_step_counter = 11
  second = gate(env, env_ids)
  assert gate.stage_index == 1
  assert second["pass_streak"].item() == 0
  _assert_stage_applied(command_cfg, _PERFORMANCE_STAGES[1])


def test_performance_gate_buckets_every_stair_terrain_and_direction():
  gate, env, command, _command_cfg, _time_out = _make_performance_gate(
    warmup_steps=0, min_stage_steps=0, required_passes=1
  )
  env_ids = torch.arange(4)
  command.backward_count[2:] = 0
  command.backward_sum[2:] = 0.0

  gate(env, env_ids)
  state = gate.state_dict()
  assert gate.stage_index == 0
  assert state["window"][_STAIR_TERRAINS[0]]["backward_tracking_count"] == 20
  assert state["window"][_STAIR_TERRAINS[1]]["backward_tracking_count"] == 0

  # Once both terrain/direction buckets are ready, a low inverse-stair score
  # fails the entire window instead of being hidden by the other stair type.
  command.backward_count[2:] = 10
  command.backward_sum[2:] = 1.0
  gate(env, env_ids)
  state = gate.state_dict()
  assert gate.stage_index == 0
  assert state["last_window_passed"] is False
  assert state["last_window"][_STAIR_TERRAINS[1]][
    "backward_tracking_sum"
  ] == pytest.approx(2.0)


def test_performance_gate_rejects_timeout_with_simultaneous_failure():
  gate, env, _command, _command_cfg, _time_out = _make_performance_gate(
    warmup_steps=0, min_stage_steps=0, required_passes=1
  )
  # One environment in each terrain bucket times out on the same step that a
  # non-timeout termination fires.  Those episodes must remain failures.
  env.termination_manager.get_term("illegal_contact")[[0, 2]] = True

  gate(env, torch.arange(4))
  state = gate.state_dict()
  assert gate.stage_index == 0
  assert state["last_window_passed"] is False
  for terrain_name in _STAIR_TERRAINS:
    bucket = state["last_window"][terrain_name]
    assert bucket["episode_count"] == 2
    assert bucket["safe_success_count"] == 1


def test_performance_gate_rejects_timeout_with_other_truncation():
  gate, env, _command, _command_cfg, _time_out = _make_performance_gate(
    warmup_steps=0, min_stage_steps=0, required_passes=1
  )
  env.termination_manager.get_term("out_of_terrain_bounds")[[0, 2]] = True

  gate(env, torch.arange(4))
  state = gate.state_dict()
  assert gate.stage_index == 0
  for terrain_name in _STAIR_TERRAINS:
    bucket = state["last_window"][terrain_name]
    assert bucket["safe_success_count"] == 1


def test_performance_gate_rejects_prolonged_single_foot_air_time():
  gate, env, _command, _command_cfg, _time_out = _make_performance_gate(
    warmup_steps=0, min_stage_steps=0, required_passes=1
  )
  foot_air_time = env.scene["feet_ground_contact"].data.current_air_time
  foot_air_time[0, 0] = 1.2
  foot_air_time[2, 3] = 1.2

  gate(env, torch.arange(4))
  state = gate.state_dict()
  assert gate.stage_index == 0
  for terrain_name in _STAIR_TERRAINS:
    bucket = state["last_window"][terrain_name]
    assert bucket["episode_count"] == 2
    assert bucket["safe_success_count"] == 1


def test_performance_gate_accepts_normal_terminal_swing_air_time():
  gate, env, _command, _command_cfg, _time_out = _make_performance_gate(
    warmup_steps=0, min_stage_steps=0, required_passes=1
  )
  env.scene["feet_ground_contact"].data.current_air_time.fill_(0.8)

  gate(env, torch.arange(4))

  assert gate.stage_index == 1


def test_performance_gate_honors_minimum_stage_dwell_and_advances_once_per_call():
  gate, env, _command, command_cfg, _time_out = _make_performance_gate(
    warmup_steps=0, min_stage_steps=100, required_passes=1
  )
  env_ids = torch.arange(4)

  env.common_step_counter = 99
  gate(env, env_ids)
  assert gate.stage_index == 0

  env.common_step_counter = 100
  gate(env, env_ids)
  assert gate.stage_index == 1
  _assert_stage_applied(command_cfg, _PERFORMANCE_STAGES[1])

  # Even with abundant level-9 evidence, one callback cannot skip stage 1.
  assert gate.stage_index != 2


def test_performance_gate_excludes_episodes_started_in_previous_stage():
  gate, env, _command, _command_cfg, _time_out = _make_performance_gate(
    warmup_steps=0,
    min_stage_steps=0,
    required_passes=1,
    min_episodes_per_terrain=1,
    min_directional_samples_per_terrain=10,
  )

  # Half the environments finish and trigger stage 1.  The other half are
  # still executing commands sampled from stage 0.
  gate(env, torch.tensor([0, 2]))
  assert gate.stage_index == 1

  gate(env, torch.tensor([1, 3]))
  state = gate.state_dict()
  assert state["window"][_STAIR_TERRAINS[0]]["episode_count"] == 0
  assert state["window"][_STAIR_TERRAINS[1]]["episode_count"] == 0

  # Once an episode actually starts and ends in stage 1, it can contribute.
  gate(env, torch.tensor([0, 2]))
  assert gate.stage_index == 2


def test_performance_gate_state_round_trip_restores_partial_window_and_stage():
  gate, env, _command, _command_cfg, _time_out = _make_performance_gate(
    warmup_steps=0, min_stage_steps=0
  )
  env.common_step_counter = 20
  gate(env, torch.arange(4))
  assert gate.state_dict()["pass_streak"] == 1

  env.common_step_counter = 21
  gate(env, torch.arange(4))
  assert gate.stage_index == 1

  # Half of the next independent window is pending at checkpoint time.
  env.common_step_counter = 22
  gate(env, torch.tensor([0, 2]))
  saved = gate.state_dict()

  restored, restored_env, restored_command, restored_cfg, _time_out = (
    _make_performance_gate(
      common_step_counter=999,
      warmup_steps=0,
      min_stage_steps=0,
    )
  )
  restored.load_state_dict(saved)
  assert restored.state_dict() == saved
  assert restored_command.restore_resample_count == 1
  _assert_stage_applied(restored_cfg, _PERFORMANCE_STAGES[1])

  restored_env.common_step_counter = 23
  restored(restored_env, torch.tensor([1, 3]))
  assert restored.stage_index == 1
  assert restored.state_dict()["pass_streak"] == 1

  restored_env.common_step_counter = 24
  restored(restored_env, torch.arange(4))
  assert restored.stage_index == 2
  _assert_stage_applied(restored_cfg, _PERFORMANCE_STAGES[2])


def test_performance_gate_load_is_atomic_on_invalid_state():
  gate, _env, _command, command_cfg, _time_out = _make_performance_gate()
  before = gate.state_dict()
  invalid = gate.state_dict()
  invalid["stage_index"] = len(_PERFORMANCE_STAGES)

  with pytest.raises(ValueError, match="stage index"):
    gate.load_state_dict(invalid)

  assert gate.state_dict() == before
  _assert_stage_applied(command_cfg, _PERFORMANCE_STAGES[0])
