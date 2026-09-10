"""Unit tests for the pure-MPC validation gate policy."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "validate_rl_mpc.py"
_SPEC = importlib.util.spec_from_file_location("mjlab_validate_rl_mpc", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DEFAULT_MOTION_SECONDS = _MODULE.DEFAULT_MOTION_SECONDS
DEFAULT_STAND_SECONDS = _MODULE.DEFAULT_STAND_SECONDS
GateThresholds = _MODULE.GateThresholds
LevelResult = _MODULE.LevelResult
_build_parser = _MODULE._build_parser
_counter_delta = _MODULE._counter_delta
_duration_for_command = _MODULE._duration_for_command
_level_failure_reasons = _MODULE._level_failure_reasons
_read_solver_failure_count = _MODULE._read_solver_failure_count
_required_episode_duration = _MODULE._required_episode_duration
_set_exact_manual_command = _MODULE._set_exact_manual_command


def _healthy_result(
  command: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> LevelResult:
  return LevelResult(
    level=0,
    command=command,
    seconds=30.0,
    termination_count=0,
    max_abs_torque=20.0,
    torque_saturation_fraction=0.0,
    min_base_height=0.27,
    max_tilt_rad=0.05,
    max_abs_roll_rad=0.04,
    max_abs_pitch_rad=0.03,
    max_abs_hip_x_rad=0.10,
    max_planar_displacement_m=0.02,
    final_planar_displacement_xy_m=((0.01, -0.01),),
    tracking_seconds=4.5,
    mean_body_velocity_xy_yaw=(command,),
    mean_abs_tracking_error_xy_yaw=((0.0, 0.0, 0.0),),
    foot_contact_fraction=(1.0, 1.0, 1.0, 1.0),
    foot_mean_contact_force_n=(35.0, 35.0, 35.0, 35.0),
    foot_peak_contact_force_n=(60.0, 60.0, 60.0, 60.0),
    nonfinite_contact_force_samples=0,
    solver_failure_count=0,
    controller_error_count=0,
    termination_counts_by_term={},
  )


def test_cli_uses_long_stand_gate_without_extending_motion_gate() -> None:
  args = _build_parser().parse_args([])

  assert args.stand_seconds == DEFAULT_STAND_SECONDS == 30.0
  assert args.seconds == DEFAULT_MOTION_SECONDS == 5.0
  assert (
    _duration_for_command((0.0, 0.0, 0.0), args.seconds, args.stand_seconds) == 30.0
  )
  assert _duration_for_command((0.5, 0.0, 0.0), args.seconds, args.stand_seconds) == 5.0


def test_required_episode_duration_covers_level_and_transition_gates() -> None:
  commands = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))

  assert (
    _required_episode_duration(
      commands=commands,
      motion_seconds=5.0,
      stand_seconds=30.0,
      validate_transitions=True,
    )
    == 30.0
  )
  assert (
    _required_episode_duration(
      commands=((0.5, 0.0, 0.0),),
      motion_seconds=8.0,
      stand_seconds=30.0,
      validate_transitions=True,
    )
    == 17.0
  )


def test_gate_rejects_a_silently_clamped_manual_command() -> None:
  command_term = SimpleNamespace(
    set_manual_command=lambda _env_idx, _command: (-0.2, 0.0, 0.0)
  )

  with pytest.raises(ValueError, match="outside the configured training envelope"):
    _set_exact_manual_command(command_term, 0, (-0.3, 0.0, 0.0))

  _set_exact_manual_command(command_term, 0, (-0.2, 0.0, 0.0))


def test_healthy_level_passes_every_nontermination_gate() -> None:
  assert _level_failure_reasons(_healthy_result(), GateThresholds()) == ()


def test_stand_metrics_fail_independently_of_termination() -> None:
  result = _healthy_result()
  result.max_planar_displacement_m = 0.08
  result.min_base_height = 0.19
  result.max_abs_roll_rad = 0.30
  result.max_abs_pitch_rad = 0.31
  result.max_abs_hip_x_rad = 0.25
  result.foot_contact_fraction = (0.97, 1.0, 1.0, 1.0)
  result.foot_mean_contact_force_n = (0.5, 35.0, 35.0, 35.0)
  result.foot_peak_contact_force_n = (60.0, 60.0, 60.0, 900.0)
  result.nonfinite_contact_force_samples = 1
  result.solver_failure_count = 2

  reasons = _level_failure_reasons(result, GateThresholds())

  assert result.termination_count == 0
  assert any("stand_drift" in reason for reason in reasons)
  assert any("min_height" in reason for reason in reasons)
  assert any("max|roll|" in reason for reason in reasons)
  assert any("max|pitch|" in reason for reason in reasons)
  assert any("max|HipX|" in reason for reason in reasons)
  assert any("FL_contact_fraction" in reason for reason in reasons)
  assert any("FL_mean_contact_force" in reason for reason in reasons)
  assert any("HR_peak_contact_force" in reason for reason in reasons)
  assert any("nonfinite_contact_forces" in reason for reason in reasons)
  assert any("solver_failures" in reason for reason in reasons)


def test_moving_command_does_not_apply_stationary_drift_limit() -> None:
  result = _healthy_result(command=(0.5, 0.0, 0.0))
  result.max_planar_displacement_m = 2.5
  result.foot_contact_fraction = (0.11, 0.11, 0.11, 0.11)

  assert _level_failure_reasons(result, GateThresholds()) == ()


def test_moving_command_requires_real_post_settling_tracking() -> None:
  linear = _healthy_result(command=(-0.2, 0.0, 0.0))
  linear.mean_body_velocity_xy_yaw = ((-0.08, 0.0, 0.0),)
  linear.mean_abs_tracking_error_xy_yaw = ((0.12, 0.0, 0.0),)
  linear_reasons = _level_failure_reasons(linear, GateThresholds())
  assert any("vx_tracking_ratio" in reason for reason in linear_reasons)

  yaw = _healthy_result(command=(0.0, 0.0, 0.5))
  yaw.mean_body_velocity_xy_yaw = ((0.0, 0.0, 0.25),)
  yaw.mean_abs_tracking_error_xy_yaw = ((0.0, 0.0, 0.25),)
  yaw_reasons = _level_failure_reasons(yaw, GateThresholds())
  assert any("wz_tracking_ratio" in reason for reason in yaw_reasons)
  assert any("wz_MAE" in reason for reason in yaw_reasons)


def test_moving_command_uses_a_separate_physical_hip_x_limit() -> None:
  result = _healthy_result(command=(0.0, 0.0, 0.5))
  result.max_abs_hip_x_rad = 0.30

  assert _level_failure_reasons(result, GateThresholds()) == ()
  result.max_abs_hip_x_rad = 0.36
  assert any(
    "max|HipX|" in reason for reason in _level_failure_reasons(result, GateThresholds())
  )


def test_missing_solver_instrumentation_is_a_gate_failure() -> None:
  result = _healthy_result()
  result.solver_failure_count = None

  assert "solver failure counter unavailable" in _level_failure_reasons(
    result, GateThresholds()
  )


def test_solver_counter_reads_all_nested_controllers() -> None:
  controllers = [
    SimpleNamespace(
      _runner=SimpleNamespace(cMPC=SimpleNamespace(mpc_solver_failure_count=2))
    ),
    SimpleNamespace(
      _runner=SimpleNamespace(cMPC=SimpleNamespace(mpc_solver_failure_count=3))
    ),
  ]
  action_term = SimpleNamespace(_controllers=controllers)

  assert _read_solver_failure_count(action_term) == 5
  assert _counter_delta(2, 5) == 3
  assert _counter_delta(4, 1) == 1


@pytest.mark.parametrize(
  "kwargs",
  [
    {"max_stationary_planar_drift_m": 0.0},
    {"min_base_height_m": 0.0},
    {"max_abs_roll_pitch_rad": float("nan")},
    {"max_moving_abs_hip_x_rad": -1.0},
    {"min_linear_tracking_ratio": 0.0},
    {"min_yaw_tracking_ratio": float("nan")},
    {"max_yaw_tracking_mae_radps": -1.0},
    {"min_standing_foot_contact_fraction": 1.1},
    {"min_mean_contact_force_n": -1.0},
    {"max_solver_failures": -1},
  ],
)
def test_gate_thresholds_reject_invalid_values(kwargs: dict[str, float]) -> None:
  with pytest.raises(ValueError):
    GateThresholds(**kwargs)
