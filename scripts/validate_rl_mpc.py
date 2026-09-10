"""Run the terrain and closed-loop pure-MPC gates before PPO training."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import dataclass

# Warp initializes its cache while importing mjlab.  Keep the validation gate
# runnable in containers and CI jobs whose user-level cache directory is read-only.
os.environ.setdefault(
  "WARP_CACHE_PATH", os.path.join(tempfile.gettempdir(), "mjlab-warp-cache")
)

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.rl_mpc.config.g23.env_cfgs import (
  syncai_g23_pure_mpc_validation_env_cfg,
)
from mjlab.tasks.rl_mpc.config.g23.terrain import (
  MAX_TERRAIN_HEIGHT_SPAN_M,
  validate_rlmpc_g23_terrain_cfg,
)
from mjlab.tasks.rl_mpc.controller.backend import MpcGaitMode
from mjlab.tasks.rl_mpc.mdp.actions import MpcFootPlacementActionCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import euler_xyz_from_quat

FOOT_NAMES = ("FL", "FR", "HL", "HR")
DEFAULT_MOTION_SECONDS = 5.0
DEFAULT_STAND_SECONDS = 30.0
TRACKING_SETTLE_SECONDS = 0.5

DEFAULT_COMMANDS = (
  (0.0, 0.0, 0.0),
  (0.5, 0.0, 0.0),
  (-0.2, 0.0, 0.0),
  (0.0, 0.15, 0.0),
  (0.0, -0.15, 0.0),
  (0.0, 0.0, 0.5),
  (0.0, 0.0, -0.5),
)


@dataclass(frozen=True, kw_only=True)
class GateThresholds:
  """Safety limits used by the pure-MPC validation gate."""

  max_stationary_planar_drift_m: float = 0.05
  min_base_height_m: float = 0.20
  max_abs_roll_pitch_rad: float = math.radians(15.0)
  # Standing should keep HipX near neutral.  Trotting needs more lateral joint
  # travel for side steps and turns, but must remain well inside the 0.523 rad
  # mechanical limit.
  max_abs_hip_x_rad: float = 0.20
  max_moving_abs_hip_x_rad: float = 0.35
  min_standing_foot_contact_fraction: float = 0.98
  min_moving_foot_contact_fraction: float = 0.10
  min_mean_contact_force_n: float = 1.0
  max_peak_contact_force_n: float = 800.0
  max_solver_failures: int = 0
  min_linear_tracking_ratio: float = 0.50
  min_yaw_tracking_ratio: float = 0.65
  max_linear_tracking_mae_mps: float = 0.15
  max_yaw_tracking_mae_radps: float = 0.18

  def __post_init__(self) -> None:
    positive_finite = {
      "max_stationary_planar_drift_m": self.max_stationary_planar_drift_m,
      "min_base_height_m": self.min_base_height_m,
      "max_abs_roll_pitch_rad": self.max_abs_roll_pitch_rad,
      "max_abs_hip_x_rad": self.max_abs_hip_x_rad,
      "max_moving_abs_hip_x_rad": self.max_moving_abs_hip_x_rad,
      "max_peak_contact_force_n": self.max_peak_contact_force_n,
      "min_linear_tracking_ratio": self.min_linear_tracking_ratio,
      "min_yaw_tracking_ratio": self.min_yaw_tracking_ratio,
      "max_linear_tracking_mae_mps": self.max_linear_tracking_mae_mps,
      "max_yaw_tracking_mae_radps": self.max_yaw_tracking_mae_radps,
    }
    for name, value in positive_finite.items():
      if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if (
      not math.isfinite(self.min_mean_contact_force_n)
      or self.min_mean_contact_force_n < 0.0
    ):
      raise ValueError("min_mean_contact_force_n must be finite and non-negative")
    for name, value in (
      (
        "min_standing_foot_contact_fraction",
        self.min_standing_foot_contact_fraction,
      ),
      ("min_moving_foot_contact_fraction", self.min_moving_foot_contact_fraction),
    ):
      if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    if (
      isinstance(self.max_solver_failures, bool)
      or not isinstance(self.max_solver_failures, int)
      or self.max_solver_failures < 0
    ):
      raise ValueError("max_solver_failures must be a non-negative integer")


@dataclass
class LevelResult:
  level: int
  command: tuple[float, float, float]
  seconds: float
  termination_count: int
  max_abs_torque: float
  torque_saturation_fraction: float
  min_base_height: float
  max_tilt_rad: float
  max_abs_roll_rad: float
  max_abs_pitch_rad: float
  max_abs_hip_x_rad: float
  max_planar_displacement_m: float
  final_planar_displacement_xy_m: tuple[tuple[float, float], ...]
  tracking_seconds: float
  mean_body_velocity_xy_yaw: tuple[tuple[float, float, float], ...]
  mean_abs_tracking_error_xy_yaw: tuple[tuple[float, float, float], ...]
  foot_contact_fraction: tuple[float, ...]
  foot_mean_contact_force_n: tuple[float, ...] | None
  foot_peak_contact_force_n: tuple[float, ...] | None
  nonfinite_contact_force_samples: int
  solver_failure_count: int | None
  controller_error_count: int
  termination_counts_by_term: dict[str, int]


@dataclass
class TransitionResult:
  level: int
  mode_sequence: tuple[str, ...]
  termination_count: int
  initial_stand_ready_by_deadline: bool
  all_envs_saw_every_mode: bool
  all_envs_finished_standing: bool
  all_feet_in_contact: bool
  final_modes: tuple[str, ...]
  final_stand_ready: tuple[bool, ...]
  final_foot_contacts: tuple[tuple[bool, ...], ...]
  solver_failure_count: int | None
  controller_error_count: int


def _is_stationary_command(command: tuple[float, float, float]) -> bool:
  return all(abs(value) <= 1e-9 for value in command)


def _duration_for_command(
  command: tuple[float, float, float],
  motion_seconds: float,
  stand_seconds: float,
) -> float:
  return stand_seconds if _is_stationary_command(command) else motion_seconds


def _required_episode_duration(
  *,
  commands: tuple[tuple[float, float, float], ...],
  motion_seconds: float,
  stand_seconds: float,
  validate_transitions: bool,
) -> float:
  level_duration = max(
    _duration_for_command(command, motion_seconds, stand_seconds)
    for command in commands
  )
  if not validate_transitions:
    return level_duration
  transition_duration = 1.0 + 2.0 * max(2.0, motion_seconds)
  return max(level_duration, transition_duration)


def _read_controller_error_count(action_term: object) -> int:
  value = getattr(action_term, "controller_error_count", None)
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise TypeError("foot_placement action has no valid controller_error_count")
  return value


def _read_solver_failure_count(action_term: object) -> int | None:
  """Return the aggregate solver count, or None when not instrumented.

  New controller revisions expose a public counter on the locomotion object.
  The direct-controller lookup keeps this validator compatible with a future
  forwarding property on the Python backend or action term.
  """
  direct = getattr(action_term, "mpc_solver_failure_count", None)
  if direct is not None:
    if isinstance(direct, bool) or not isinstance(direct, int) or direct < 0:
      raise TypeError("mpc_solver_failure_count must be a non-negative integer")
    return direct

  controllers = getattr(action_term, "_controllers", None)
  if not isinstance(controllers, list) or not controllers:
    return None
  counts: list[int] = []
  for controller in controllers:
    sources = (
      controller,
      getattr(getattr(controller, "_runner", None), "cMPC", None),
    )
    value = None
    for source in sources:
      if source is None:
        continue
      value = getattr(source, "mpc_solver_failure_count", None)
      if value is None:
        value = getattr(source, "_mpc_solver_failure_count", None)
      if value is not None:
        break
    if value is None:
      return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
      raise TypeError("mpc_solver_failure_count must be a non-negative integer")
    counts.append(value)
  return sum(counts)


def _counter_delta(before: int | None, after: int | None) -> int | None:
  if before is None or after is None:
    return None
  if after < before:
    # Controller reset may intentionally clear an episode-local counter.
    return after
  return after - before


def _place_all_terrain_types_at_level(env: ManagerBasedRlEnv, level: int) -> None:
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  if not 0 <= level < terrain.terrain_origins.shape[0]:
    raise ValueError(f"Terrain level {level} is outside the generated bank")
  terrain.terrain_levels.fill_(level)
  terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]


def _set_exact_manual_command(
  command_term: UniformVelocityCommand,
  env_idx: int,
  command: tuple[float, float, float],
) -> None:
  """Apply a gate command without silently validating a clamped substitute."""
  applied = command_term.set_manual_command(env_idx, command)
  if applied is None or any(
    not math.isclose(requested, actual, rel_tol=0.0, abs_tol=1e-9)
    for requested, actual in zip(command, applied, strict=True)
  ):
    raise ValueError(
      f"Requested gate command {command} is outside the configured training "
      f"envelope; command term applied {applied}."
    )


def _run_level(
  env: ManagerBasedRlEnv,
  level: int,
  seconds: float,
  command: tuple[float, float, float],
) -> LevelResult:
  _place_all_terrain_types_at_level(env, level)
  env.reset()
  command_term = env.command_manager.get_term("twist")
  assert isinstance(command_term, UniformVelocityCommand)
  for env_idx in range(env.num_envs):
    _set_exact_manual_command(command_term, env_idx, command)

  action_term = env.action_manager.get_term("foot_placement")
  solver_failures_before = _read_solver_failure_count(action_term)
  controller_errors_before = _read_controller_error_count(action_term)
  robot = env.scene["robot"]
  hip_x_ids, hip_x_names = robot.find_joints((r".*_HipX_joint",), preserve_order=True)
  if len(hip_x_ids) != 4:
    raise RuntimeError(f"Expected four G23 HipX joints, got {tuple(hip_x_names)}")
  hip_x_ids_tensor = torch.tensor(hip_x_ids, device=env.device, dtype=torch.long)

  contact_sensor = env.scene["feet_ground_contact"]
  found = contact_sensor.data.found
  if found is None or found.shape[0] != env.num_envs or found.shape[1] % 4 != 0:
    raise RuntimeError(
      "feet_ground_contact must expose primary-major found data for four feet"
    )
  contact_slots = found.shape[1] // 4
  has_force_data = contact_sensor.data.force is not None

  steps = math.ceil(seconds / env.step_dt)
  action = torch.zeros((env.num_envs, 8), device=env.device)
  termination_count = 0
  termination_counts_by_term = {
    name: 0 for name in env.termination_manager.active_terms
  }
  saturated = 0
  torque_samples = 0
  max_abs_torque = 0.0
  min_base_height = float("inf")
  max_tilt_rad = 0.0
  max_abs_roll_rad = 0.0
  max_abs_pitch_rad = 0.0
  max_abs_hip_x_rad = 0.0
  max_planar_displacement_m = 0.0
  initial_planar_position = robot.data.root_link_pos_w[:, :2].clone()
  contact_counts = torch.zeros(
    (env.num_envs, 4), device=env.device, dtype=torch.float64
  )
  force_sum = torch.zeros_like(contact_counts)
  force_contact_counts = torch.zeros_like(contact_counts)
  peak_force = torch.zeros_like(contact_counts)
  nonfinite_contact_force_samples = 0
  tracking_velocity_sum = torch.zeros(
    (env.num_envs, 3), device=env.device, dtype=torch.float64
  )
  tracking_abs_error_sum = torch.zeros_like(tracking_velocity_sum)
  command_tensor = torch.tensor(command, device=env.device, dtype=torch.float64)
  tracking_start_step = min(
    math.ceil(TRACKING_SETTLE_SECONDS / env.step_dt), max(0, steps - 1)
  )
  tracking_samples = 0

  elapsed_steps = 0
  for step in range(steps):
    _, _, terminated, _, _ = env.step(action)
    elapsed_steps = step + 1
    termination_count += int(torch.count_nonzero(terminated).item())
    for name in termination_counts_by_term:
      termination_counts_by_term[name] += int(
        torch.count_nonzero(env.termination_manager.get_term(name)).item()
      )

    torque = robot.data.qfrc_actuator[:, :12]
    limits = torch.tensor((24.0, 24.0, 36.0) * 4, device=env.device, dtype=torque.dtype)
    max_abs_torque = max(max_abs_torque, float(torch.max(torch.abs(torque)).item()))
    saturated += int(torch.count_nonzero(torch.abs(torque) >= limits * 0.999).item())
    torque_samples += torque.numel()

    local_height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    min_base_height = min(min_base_height, float(torch.min(local_height).item()))
    projected_gravity = robot.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))
    max_tilt_rad = max(max_tilt_rad, float(torch.max(tilt).item()))
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
    max_abs_roll_rad = max(max_abs_roll_rad, float(torch.max(torch.abs(roll)).item()))
    max_abs_pitch_rad = max(
      max_abs_pitch_rad, float(torch.max(torch.abs(pitch)).item())
    )
    max_abs_hip_x_rad = max(
      max_abs_hip_x_rad,
      float(torch.max(torch.abs(robot.data.joint_pos[:, hip_x_ids_tensor])).item()),
    )

    planar_displacement = robot.data.root_link_pos_w[:, :2] - initial_planar_position
    max_planar_displacement_m = max(
      max_planar_displacement_m,
      float(torch.max(torch.linalg.vector_norm(planar_displacement, dim=-1)).item()),
    )

    if step >= tracking_start_step:
      body_velocity = torch.column_stack(
        (
          robot.data.root_link_lin_vel_b[:, 0],
          robot.data.root_link_lin_vel_b[:, 1],
          robot.data.root_link_ang_vel_b[:, 2],
        )
      ).to(torch.float64)
      tracking_velocity_sum += body_velocity
      tracking_abs_error_sum += torch.abs(body_velocity - command_tensor)
      tracking_samples += 1

    found = contact_sensor.data.found
    if found is None:
      raise RuntimeError("feet_ground_contact lost its found data")
    foot_contacts = found.reshape(env.num_envs, 4, contact_slots).gt(0).any(dim=-1)
    contact_counts += foot_contacts.to(contact_counts.dtype)
    force_data = contact_sensor.data.force
    if has_force_data:
      if force_data is None:
        raise RuntimeError("feet_ground_contact lost its force data")
      reshaped_force = force_data.reshape(env.num_envs, 4, contact_slots, 3)
      foot_force = torch.linalg.vector_norm(reshaped_force, dim=-1).sum(dim=-1)
      finite_force = torch.isfinite(foot_force)
      nonfinite_contact_force_samples += int(torch.count_nonzero(~finite_force).item())
      safe_force = torch.where(finite_force, foot_force, 0.0).to(torch.float64)
      valid_contact_force = foot_contacts & finite_force
      force_sum += torch.where(valid_contact_force, safe_force, 0.0)
      force_contact_counts += valid_contact_force.to(torch.float64)
      peak_force = torch.maximum(peak_force, safe_force)

    # Validation disables automatic reset. Stop on the true first failure so
    # stale post-terminal state cannot hide which command/terrain failed.
    if torch.any(terminated):
      break

  final_planar_displacement = (
    robot.data.root_link_pos_w[:, :2] - initial_planar_position
  )
  tracking_divisor = max(tracking_samples, 1)
  mean_body_velocity = tracking_velocity_sum / tracking_divisor
  mean_abs_tracking_error = tracking_abs_error_sum / tracking_divisor
  contact_fraction = contact_counts / max(elapsed_steps, 1)
  min_contact_fraction_by_foot = torch.amin(contact_fraction, dim=0)
  mean_force_by_foot: tuple[float, ...] | None = None
  peak_force_by_foot: tuple[float, ...] | None = None
  if has_force_data:
    mean_force = force_sum / torch.clamp(force_contact_counts, min=1.0)
    mean_force_by_foot = tuple(
      float(value) for value in torch.amin(mean_force, dim=0).detach().cpu().tolist()
    )
    peak_force_by_foot = tuple(
      float(value) for value in torch.amax(peak_force, dim=0).detach().cpu().tolist()
    )
  solver_failures_after = _read_solver_failure_count(action_term)
  controller_errors_after = _read_controller_error_count(action_term)

  return LevelResult(
    level=level,
    command=command,
    seconds=elapsed_steps * env.step_dt,
    termination_count=termination_count,
    max_abs_torque=max_abs_torque,
    torque_saturation_fraction=saturated / max(torque_samples, 1),
    min_base_height=min_base_height,
    max_tilt_rad=max_tilt_rad,
    max_abs_roll_rad=max_abs_roll_rad,
    max_abs_pitch_rad=max_abs_pitch_rad,
    max_abs_hip_x_rad=max_abs_hip_x_rad,
    max_planar_displacement_m=max_planar_displacement_m,
    final_planar_displacement_xy_m=tuple(
      (float(row[0]), float(row[1]))
      for row in final_planar_displacement.detach().cpu().tolist()
    ),
    tracking_seconds=tracking_samples * env.step_dt,
    mean_body_velocity_xy_yaw=tuple(
      (float(row[0]), float(row[1]), float(row[2]))
      for row in mean_body_velocity.detach().cpu().tolist()
    ),
    mean_abs_tracking_error_xy_yaw=tuple(
      (float(row[0]), float(row[1]), float(row[2]))
      for row in mean_abs_tracking_error.detach().cpu().tolist()
    ),
    foot_contact_fraction=tuple(
      float(value) for value in min_contact_fraction_by_foot.detach().cpu().tolist()
    ),
    foot_mean_contact_force_n=mean_force_by_foot,
    foot_peak_contact_force_n=peak_force_by_foot,
    nonfinite_contact_force_samples=nonfinite_contact_force_samples,
    solver_failure_count=_counter_delta(solver_failures_before, solver_failures_after),
    controller_error_count=max(0, controller_errors_after - controller_errors_before),
    termination_counts_by_term=termination_counts_by_term,
  )


def _run_transition_gate(
  env: ManagerBasedRlEnv,
  level: int,
  phase_seconds: float,
) -> TransitionResult:
  """Check STAND -> TROT -> STOPPING -> STAND without resetting the episode."""
  _place_all_terrain_types_at_level(env, level)
  env.reset()
  command_term = env.command_manager.get_term("twist")
  assert isinstance(command_term, UniformVelocityCommand)
  action_term = env.action_manager.get_term("foot_placement")
  solver_failures_before = _read_solver_failure_count(action_term)
  controller_errors_before = _read_controller_error_count(action_term)
  gait_modes = getattr(action_term, "gait_modes", None)
  if not isinstance(gait_modes, torch.Tensor):
    raise TypeError("foot_placement action does not expose gait_modes")
  stand_ready = getattr(action_term, "stand_ready", None)
  if not isinstance(stand_ready, torch.Tensor):
    raise TypeError("foot_placement action does not expose stand_ready")

  action = torch.zeros((env.num_envs, 8), device=env.device)
  seen_modes = [{int(mode)} for mode in gait_modes.detach().cpu().tolist()]
  mode_sequence = [MpcGaitMode(int(gait_modes[0].item())).name]
  termination_count = 0
  initial_stand_ready_by_deadline = False
  # Give initial standing one second to settle. Moving and final stopping use
  # at least two seconds so this safety gate remains meaningful even when a
  # caller requests a short per-command smoke test.
  schedule = (
    ((0.0, 0.0, 0.0), 1.0),
    ((0.5, 0.0, 0.0), max(2.0, phase_seconds)),
    ((0.0, 0.0, 0.0), max(2.0, phase_seconds)),
  )
  for phase_idx, (command, seconds) in enumerate(schedule):
    for env_idx in range(env.num_envs):
      _set_exact_manual_command(command_term, env_idx, command)
    for step_idx in range(math.ceil(seconds / env.step_dt)):
      _, _, terminated, _, _ = env.step(action)
      termination_count += int(torch.count_nonzero(terminated).item())
      modes = gait_modes.detach().cpu().tolist()
      for env_idx, mode in enumerate(modes):
        seen_modes[env_idx].add(int(mode))
      mode_name = MpcGaitMode(int(modes[0])).name
      if mode_name != mode_sequence[-1]:
        mode_sequence.append(mode_name)
      if (
        phase_idx == 0
        and step_idx < math.ceil(0.20 / env.step_dt)
        and torch.all(stand_ready)
      ):
        initial_stand_ready_by_deadline = True
      if torch.any(terminated):
        break
    if termination_count:
      break

  expected_modes = {int(mode) for mode in MpcGaitMode}
  all_envs_saw_every_mode = all(
    expected_modes.issubset(env_modes) for env_modes in seen_modes
  )
  all_envs_finished_standing = bool(
    torch.all(gait_modes == int(MpcGaitMode.STAND)).item()
    and torch.all(stand_ready).item()
  )
  contact_sensor = env.scene["feet_ground_contact"]
  found = contact_sensor.data.found
  if found is None:
    raise RuntimeError("feet_ground_contact does not provide found data")
  all_feet_in_contact = bool(
    torch.all(found.reshape(env.num_envs, 4, -1).gt(0).any(dim=-1)).item()
  )
  final_contacts = found.reshape(env.num_envs, 4, -1).gt(0).any(dim=-1)
  solver_failures_after = _read_solver_failure_count(action_term)
  controller_errors_after = _read_controller_error_count(action_term)
  return TransitionResult(
    level=level,
    mode_sequence=tuple(mode_sequence),
    termination_count=termination_count,
    initial_stand_ready_by_deadline=initial_stand_ready_by_deadline,
    all_envs_saw_every_mode=all_envs_saw_every_mode,
    all_envs_finished_standing=all_envs_finished_standing,
    all_feet_in_contact=all_feet_in_contact,
    final_modes=tuple(
      MpcGaitMode(int(mode)).name for mode in gait_modes.detach().cpu().tolist()
    ),
    final_stand_ready=tuple(
      bool(value) for value in stand_ready.detach().cpu().tolist()
    ),
    final_foot_contacts=tuple(
      tuple(bool(value) for value in row)
      for row in final_contacts.detach().cpu().tolist()
    ),
    solver_failure_count=_counter_delta(solver_failures_before, solver_failures_after),
    controller_error_count=max(0, controller_errors_after - controller_errors_before),
  )


def _format_foot_values(
  values: tuple[float, ...] | None,
  *,
  precision: int,
  suffix: str = "",
) -> str:
  if values is None:
    return "unavailable"
  return " ".join(
    f"{foot}={value:.{precision}f}{suffix}"
    for foot, value in zip(FOOT_NAMES, values, strict=True)
  )


def _level_failure_reasons(
  result: LevelResult,
  thresholds: GateThresholds,
) -> tuple[str, ...]:
  reasons: list[str] = []
  if result.termination_count:
    reasons.append(f"terminations={result.termination_count}")
  if result.controller_error_count:
    reasons.append(f"controller_errors={result.controller_error_count}")
  if result.solver_failure_count is None:
    reasons.append("solver failure counter unavailable")
  elif result.solver_failure_count > thresholds.max_solver_failures:
    reasons.append(
      f"solver_failures={result.solver_failure_count} "
      f"> {thresholds.max_solver_failures}"
    )
  if _is_stationary_command(result.command) and (
    result.max_planar_displacement_m > thresholds.max_stationary_planar_drift_m
  ):
    reasons.append(
      f"stand_drift={result.max_planar_displacement_m:.3f}m "
      f"> {thresholds.max_stationary_planar_drift_m:.3f}m"
    )
  if not _is_stationary_command(result.command):
    if result.tracking_seconds <= 0.0:
      reasons.append("no post-settling tracking samples")
    axis_names = ("vx", "vy", "wz")
    for env_idx, (mean_velocity, mean_abs_error) in enumerate(
      zip(
        result.mean_body_velocity_xy_yaw,
        result.mean_abs_tracking_error_xy_yaw,
        strict=True,
      )
    ):
      for axis, (axis_name, requested) in enumerate(
        zip(axis_names, result.command, strict=True)
      ):
        if abs(requested) <= 1e-9:
          continue
        observed = mean_velocity[axis]
        error = mean_abs_error[axis]
        if not math.isfinite(observed) or not math.isfinite(error):
          reasons.append(f"env{env_idx}_{axis_name}_tracking_nonfinite")
          continue
        tracking_ratio = observed / requested
        ratio_limit = (
          thresholds.min_yaw_tracking_ratio
          if axis == 2
          else thresholds.min_linear_tracking_ratio
        )
        if tracking_ratio < ratio_limit:
          reasons.append(
            f"env{env_idx}_{axis_name}_tracking_ratio={tracking_ratio:.3f} "
            f"< {ratio_limit:.3f}"
          )
        error_limit = (
          thresholds.max_yaw_tracking_mae_radps
          if axis == 2
          else thresholds.max_linear_tracking_mae_mps
        )
        if error > error_limit:
          unit = "rad/s" if axis == 2 else "m/s"
          reasons.append(
            f"env{env_idx}_{axis_name}_MAE={error:.3f}{unit} > {error_limit:.3f}{unit}"
          )
  if result.min_base_height < thresholds.min_base_height_m:
    reasons.append(
      f"min_height={result.min_base_height:.3f}m < {thresholds.min_base_height_m:.3f}m"
    )
  if result.max_abs_roll_rad > thresholds.max_abs_roll_pitch_rad:
    reasons.append(
      f"max|roll|={math.degrees(result.max_abs_roll_rad):.1f}deg "
      f"> {math.degrees(thresholds.max_abs_roll_pitch_rad):.1f}deg"
    )
  if result.max_abs_pitch_rad > thresholds.max_abs_roll_pitch_rad:
    reasons.append(
      f"max|pitch|={math.degrees(result.max_abs_pitch_rad):.1f}deg "
      f"> {math.degrees(thresholds.max_abs_roll_pitch_rad):.1f}deg"
    )
  hip_x_limit = (
    thresholds.max_abs_hip_x_rad
    if _is_stationary_command(result.command)
    else thresholds.max_moving_abs_hip_x_rad
  )
  if result.max_abs_hip_x_rad > hip_x_limit:
    reasons.append(
      f"max|HipX|={result.max_abs_hip_x_rad:.3f}rad > {hip_x_limit:.3f}rad"
    )

  min_contact_fraction = (
    thresholds.min_standing_foot_contact_fraction
    if _is_stationary_command(result.command)
    else thresholds.min_moving_foot_contact_fraction
  )
  for foot, fraction in zip(FOOT_NAMES, result.foot_contact_fraction, strict=True):
    if fraction < min_contact_fraction:
      reasons.append(
        f"{foot}_contact_fraction={fraction:.3f} < {min_contact_fraction:.3f}"
      )

  if result.nonfinite_contact_force_samples:
    reasons.append(f"nonfinite_contact_forces={result.nonfinite_contact_force_samples}")
  if result.foot_mean_contact_force_n is not None:
    for foot, force in zip(FOOT_NAMES, result.foot_mean_contact_force_n, strict=True):
      if force < thresholds.min_mean_contact_force_n:
        reasons.append(
          f"{foot}_mean_contact_force={force:.1f}N "
          f"< {thresholds.min_mean_contact_force_n:.1f}N"
        )
  if result.foot_peak_contact_force_n is not None:
    for foot, force in zip(FOOT_NAMES, result.foot_peak_contact_force_n, strict=True):
      if force > thresholds.max_peak_contact_force_n:
        reasons.append(
          f"{foot}_peak_contact_force={force:.1f}N "
          f"> {thresholds.max_peak_contact_force_n:.1f}N"
        )
  return tuple(reasons)


def validate_pure_mpc(
  *,
  levels: tuple[int, ...],
  seconds: float,
  stand_seconds: float = DEFAULT_STAND_SECONDS,
  commands: tuple[tuple[float, float, float], ...],
  device: str,
  validate_transitions: bool = True,
  thresholds: GateThresholds | None = None,
) -> list[LevelResult]:
  if thresholds is None:
    thresholds = GateThresholds()
  cfg = syncai_g23_pure_mpc_validation_env_cfg()
  cfg.auto_reset = False
  # Keep controller errors observable instead of aborting on the first solver
  # exception.  Both error counters remain hard gate failures below.
  action_cfg = cfg.actions.get("foot_placement")
  if not isinstance(action_cfg, MpcFootPlacementActionCfg):
    raise TypeError("foot_placement action config cannot report controller errors")
  action_cfg.fail_on_controller_error = False
  if action_cfg.terrain_scan_miss_policy != "error":
    raise RuntimeError("Pure MPC validation must use strict terrain scan errors")

  required_duration = _required_episode_duration(
    commands=commands,
    motion_seconds=seconds,
    stand_seconds=stand_seconds,
    validate_transitions=validate_transitions,
  )
  # The deterministic validation config historically used a 20 s episode.
  # Add a full-second margin so a 30 s stand gate cannot time out on its last step.
  cfg.episode_length_s = max(cfg.episode_length_s, required_duration + 1.0)
  terrain_cfg = cfg.scene.terrain
  assert terrain_cfg is not None and terrain_cfg.terrain_generator is not None
  validate_rlmpc_g23_terrain_cfg(terrain_cfg.terrain_generator)
  print(
    "[PASS] terrain contract: flat + continuous uneven, "
    f"peak-to-peak <= {MAX_TERRAIN_HEIGHT_SPAN_M:.2f} m, no stairs"
  )
  print("[INFO] terrain scan: strict error on any missed ray")
  print(
    "[INFO] gate durations: "
    f"stand={stand_seconds:.2f}s motion={seconds:.2f}s; "
    "limits: "
    f"stand_drift<={thresholds.max_stationary_planar_drift_m:.3f}m, "
    f"height>={thresholds.min_base_height_m:.3f}m, "
    "|roll/pitch|<="
    f"{math.degrees(thresholds.max_abs_roll_pitch_rad):.1f}deg, "
    f"|HipX|<=stand:{thresholds.max_abs_hip_x_rad:.3f}/"
    f"move:{thresholds.max_moving_abs_hip_x_rad:.3f}rad, "
    f"tracking_ratio>=linear:{thresholds.min_linear_tracking_ratio:.2f}/"
    f"yaw:{thresholds.min_yaw_tracking_ratio:.2f}, "
    f"tracking_MAE<=linear:{thresholds.max_linear_tracking_mae_mps:.2f}m/s/"
    f"yaw:{thresholds.max_yaw_tracking_mae_radps:.2f}rad/s, "
    f"solver_failures<={thresholds.max_solver_failures}"
  )

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    results = [
      _run_level(
        env,
        level,
        _duration_for_command(command, seconds, stand_seconds),
        command,
      )
      for command in commands
      for level in levels
    ]
    transition_results = (
      [_run_transition_gate(env, level, seconds) for level in levels]
      if validate_transitions
      else []
    )
  finally:
    env.close()

  failed_commands: list[tuple[LevelResult, tuple[str, ...]]] = []
  for result in results:
    failure_reasons = _level_failure_reasons(result, thresholds)
    status = "FAIL" if failure_reasons else "PASS"
    solver_failure_text = (
      "unavailable"
      if result.solver_failure_count is None
      else str(result.solver_failure_count)
    )
    print(
      f"[{status}] command={result.command} level={result.level} "
      f"seconds={result.seconds:.2f} "
      f"terminations={result.termination_count} "
      f"solver_failures={solver_failure_text} "
      f"controller_errors={result.controller_error_count} "
      f"max|tau|={result.max_abs_torque:.2f}Nm "
      f"saturation={100 * result.torque_saturation_fraction:.3f}% "
      f"min_height={result.min_base_height:.3f}m "
      f"max|roll|={math.degrees(result.max_abs_roll_rad):.1f}deg "
      f"max|pitch|={math.degrees(result.max_abs_pitch_rad):.1f}deg "
      f"max|HipX|={result.max_abs_hip_x_rad:.3f}rad "
      f"max_planar_displacement={result.max_planar_displacement_m:.3f}m "
      f"terms={result.termination_counts_by_term}"
    )
    print(
      "       contact_fraction(min/env): "
      f"{_format_foot_values(result.foot_contact_fraction, precision=3)}"
    )
    print(
      "       mean_contact_force(min/env): "
      f"{_format_foot_values(result.foot_mean_contact_force_n, precision=1, suffix='N')}"
    )
    print(
      "       peak_contact_force(max/env): "
      f"{_format_foot_values(result.foot_peak_contact_force_n, precision=1, suffix='N')}"
    )
    print(
      "       final_xy_by_env="
      + " ".join(
        f"env{env_idx}=({x:+.3f},{y:+.3f})m"
        for env_idx, (x, y) in enumerate(result.final_planar_displacement_xy_m)
      )
    )
    print(
      f"       tracking_after_settle={result.tracking_seconds:.2f}s "
      "mean_(vx,vy,wz)_by_env="
      + " ".join(
        f"env{env_idx}=({vx:+.3f},{vy:+.3f},{wz:+.3f})"
        for env_idx, (vx, vy, wz) in enumerate(result.mean_body_velocity_xy_yaw)
      )
    )
    print(
      "       mean_abs_tracking_error_(vx,vy,wz)_by_env="
      + " ".join(
        f"env{env_idx}=({vx:.3f},{vy:.3f},{wz:.3f})"
        for env_idx, (vx, vy, wz) in enumerate(result.mean_abs_tracking_error_xy_yaw)
      )
    )
    if failure_reasons:
      print("       violations: " + "; ".join(failure_reasons))
      failed_commands.append((result, failure_reasons))
  failed_transitions: list[TransitionResult] = []
  for result in transition_results:
    transition_ok = (
      result.termination_count == 0
      and result.controller_error_count == 0
      and result.solver_failure_count is not None
      and result.solver_failure_count <= thresholds.max_solver_failures
      and result.initial_stand_ready_by_deadline
      and result.all_envs_saw_every_mode
      and result.all_envs_finished_standing
      and result.all_feet_in_contact
    )
    solver_failure_text = (
      "unavailable"
      if result.solver_failure_count is None
      else str(result.solver_failure_count)
    )
    print(
      f"[{'PASS' if transition_ok else 'FAIL'}] transition level={result.level} "
      f"modes={' -> '.join(result.mode_sequence)} "
      f"terminations={result.termination_count} "
      f"solver_failures={solver_failure_text} "
      f"controller_errors={result.controller_error_count} "
      f"initial_ready={result.initial_stand_ready_by_deadline} "
      f"all_modes={result.all_envs_saw_every_mode} "
      f"finished_standing={result.all_envs_finished_standing} "
      f"all_feet_contact={result.all_feet_in_contact} "
      f"final_modes={result.final_modes} "
      f"final_ready={result.final_stand_ready} "
      f"final_contacts={result.final_foot_contacts}"
    )
    if not transition_ok:
      failed_transitions.append(result)
  if failed_commands or failed_transitions:
    failures = [
      f"command={result.command}, level={result.level}: {', '.join(reasons)}"
      for result, reasons in failed_commands
    ]
    failures.extend(f"start/stop level={result.level}" for result in failed_transitions)
    raise RuntimeError("Pure MPC gate failed: " + "; ".join(failures))
  print("[PASS] pure MPC closed-loop gate")
  return results


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Validate terrain, 30-second stationary stability, training-envelope "
      "commands, and same-episode MPC mode transitions before PPO training."
    )
  )
  parser.add_argument("--levels", type=int, nargs="+", default=[0, 5])
  parser.add_argument(
    "--seconds",
    type=float,
    default=DEFAULT_MOTION_SECONDS,
    help="Seconds per non-zero command and moving/stopping transition phase.",
  )
  parser.add_argument(
    "--stand-seconds",
    type=float,
    default=DEFAULT_STAND_SECONDS,
    help="Seconds for the zero-command stationary stability gate (default: 30).",
  )
  parser.add_argument(
    "--command",
    dest="commands",
    type=float,
    nargs=3,
    action="append",
    help=(
      "Body-frame vx vy wz command. Repeat for a custom matrix; by default the "
      "zero command and six one-axis training-envelope limits are checked."
    ),
  )
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--skip-transition",
    action="store_true",
    help="Skip the same-episode stand/start/stop gate (enabled by default).",
  )
  parser.add_argument(
    "--max-stand-drift",
    type=float,
    default=GateThresholds.max_stationary_planar_drift_m,
    metavar="METRES",
  )
  parser.add_argument(
    "--min-base-height",
    type=float,
    default=GateThresholds.min_base_height_m,
    metavar="METRES",
  )
  parser.add_argument(
    "--max-roll-pitch-deg",
    type=float,
    default=math.degrees(GateThresholds.max_abs_roll_pitch_rad),
    metavar="DEGREES",
  )
  parser.add_argument(
    "--max-hip-x",
    type=float,
    default=GateThresholds.max_abs_hip_x_rad,
    metavar="RADIANS",
  )
  parser.add_argument(
    "--max-moving-hip-x",
    type=float,
    default=GateThresholds.max_moving_abs_hip_x_rad,
    metavar="RADIANS",
  )
  parser.add_argument(
    "--min-stand-contact-fraction",
    type=float,
    default=GateThresholds.min_standing_foot_contact_fraction,
  )
  parser.add_argument(
    "--min-moving-contact-fraction",
    type=float,
    default=GateThresholds.min_moving_foot_contact_fraction,
  )
  parser.add_argument(
    "--min-mean-contact-force",
    type=float,
    default=GateThresholds.min_mean_contact_force_n,
    metavar="NEWTONS",
  )
  parser.add_argument(
    "--max-peak-contact-force",
    type=float,
    default=GateThresholds.max_peak_contact_force_n,
    metavar="NEWTONS",
  )
  parser.add_argument(
    "--max-solver-failures",
    type=int,
    default=GateThresholds.max_solver_failures,
  )
  parser.add_argument(
    "--min-linear-tracking-ratio",
    type=float,
    default=GateThresholds.min_linear_tracking_ratio,
    help="Minimum post-settling mean vx/vy divided by its non-zero command.",
  )
  parser.add_argument(
    "--min-yaw-tracking-ratio",
    type=float,
    default=GateThresholds.min_yaw_tracking_ratio,
    help="Minimum post-settling mean yaw rate divided by its non-zero command.",
  )
  parser.add_argument(
    "--max-linear-tracking-mae",
    type=float,
    default=GateThresholds.max_linear_tracking_mae_mps,
    metavar="M/S",
  )
  parser.add_argument(
    "--max-yaw-tracking-mae",
    type=float,
    default=GateThresholds.max_yaw_tracking_mae_radps,
    metavar="RAD/S",
  )
  return parser


def main() -> None:
  parser = _build_parser()
  args = parser.parse_args()
  if not math.isfinite(args.seconds) or args.seconds <= 0.0:
    raise ValueError("--seconds must be finite and positive")
  if not math.isfinite(args.stand_seconds) or args.stand_seconds <= 0.0:
    raise ValueError("--stand-seconds must be finite and positive")
  commands = (
    DEFAULT_COMMANDS
    if args.commands is None
    else tuple(tuple(command) for command in args.commands)
  )
  if not commands or not all(
    len(command) == 3 and all(math.isfinite(value) for value in command)
    for command in commands
  ):
    raise ValueError("--command values must be finite vx vy wz triples")
  thresholds = GateThresholds(
    max_stationary_planar_drift_m=args.max_stand_drift,
    min_base_height_m=args.min_base_height,
    max_abs_roll_pitch_rad=math.radians(args.max_roll_pitch_deg),
    max_abs_hip_x_rad=args.max_hip_x,
    max_moving_abs_hip_x_rad=args.max_moving_hip_x,
    min_standing_foot_contact_fraction=args.min_stand_contact_fraction,
    min_moving_foot_contact_fraction=args.min_moving_contact_fraction,
    min_mean_contact_force_n=args.min_mean_contact_force,
    max_peak_contact_force_n=args.max_peak_contact_force,
    max_solver_failures=args.max_solver_failures,
    min_linear_tracking_ratio=args.min_linear_tracking_ratio,
    min_yaw_tracking_ratio=args.min_yaw_tracking_ratio,
    max_linear_tracking_mae_mps=args.max_linear_tracking_mae,
    max_yaw_tracking_mae_radps=args.max_yaw_tracking_mae,
  )
  validate_pure_mpc(
    levels=tuple(args.levels),
    seconds=args.seconds,
    stand_seconds=args.stand_seconds,
    commands=commands,
    device=args.device,
    validate_transitions=not args.skip_transition,
    thresholds=thresholds,
  )


if __name__ == "__main__":
  main()
