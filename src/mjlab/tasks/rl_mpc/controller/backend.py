"""Lazy adapter for the external rl-mpc-locomotion controller.

The task registry must remain importable when the optional native solver is not
built.  External imports therefore happen only when an environment constructs
its action term.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

CONTROLLER_ROOT_ENV = "MJLAB_RLMPC_CONTROLLER_ROOT"


class MpcGaitMode(IntEnum):
  """Observation contract shared with the legacy controller."""

  TROT = 0
  STOPPING = 1
  STAND = 2


NUM_MPC_GAIT_MODES = len(MpcGaitMode)


class MpcController(Protocol):
  """Minimal controller interface used by the mjlab action term."""

  def reset(self) -> None: ...

  @property
  def gait_mode(self) -> int: ...

  @property
  def stand_ready(self) -> bool: ...

  @property
  def mpc_solver_failure_count(self) -> int: ...

  @property
  def observation_snapshot(self) -> np.ndarray: ...

  def step(
    self,
    dof_states: np.ndarray,
    body_states: np.ndarray,
    command: np.ndarray,
    foot_placement_offsets: np.ndarray,
    foot_contacts: np.ndarray | None = None,
    terrain_height_samples: np.ndarray | None = None,
  ) -> np.ndarray: ...


@dataclass(frozen=True, kw_only=True)
class LegacyMpcBackendCfg:
  """Configuration for the local rl-mpc-locomotion controller."""

  controller_root: str | None = None
  formulation: Literal["native", "syncai"] = "syncai"
  controller_dt: float = 0.01
  iterations_between_mpc: int = 2
  gait_period: float | None = None
  flat_ground: bool = False
  body_mass: float | None = None
  foot_landing_height: float | None = None
  enable_stand_mode: bool = False
  stand_enter_linear_speed: float = 0.03
  stand_enter_yaw_rate: float = 0.05
  stand_exit_linear_speed: float = 0.08
  stand_exit_yaw_rate: float = 0.10
  stand_command_hold_time: float = 0.20
  stand_contact_hold_time: float = 0.05
  stand_contact_loss_time: float = 0.05
  stand_arm_timeout: float = 0.20
  stand_max_linear_speed: float = 0.10
  stand_max_vertical_speed: float = 0.10
  stand_max_yaw_rate: float = 0.20
  stand_max_roll_pitch_rate: float = 0.20
  stand_foot_search_rate: float = 0.05
  stand_foot_search_depth: float = 0.08
  stand_mpc_xy_position_weight: float = 5.0
  stand_mpc_vxy_weight: float = 3.0
  stand_kp_cartesian: tuple[float, float, float] = (80.0, 80.0, 30.0)
  stand_kd_cartesian: tuple[float, float, float] = (8.0, 8.0, 8.0)
  stand_hipx_kp: float = 12.0
  stand_hipx_kd: float = 1.0
  symmetric_residuals: bool = False
  stop_reposition_steps: int = 0


def _development_controller_root() -> Path | None:
  """Find the adjacent controller checkout used by this development workspace."""
  for parent in Path(__file__).resolve().parents:
    if parent.name == "mjlab_test":
      candidate = parent.parent / "rl-mpc-locomotion"
      if (candidate / "MPC_Controller").is_dir():
        return candidate
      break
  return None


def resolve_controller_root(explicit_root: str | None = None) -> Path | None:
  """Resolve an external controller checkout, or return None for an installed one."""
  raw_root = explicit_root or os.environ.get(CONTROLLER_ROOT_ENV)
  if raw_root:
    root = Path(raw_root).expanduser().resolve()
    if not (root / "MPC_Controller").is_dir():
      raise FileNotFoundError(f"No MPC_Controller package under {root}")
    return root

  try:
    if importlib.util.find_spec("MPC_Controller") is not None:
      return None
  except (ImportError, ValueError):
    pass
  return _development_controller_root()


def _load_external_modules(cfg: LegacyMpcBackendCfg):
  root = resolve_controller_root(cfg.controller_root)
  if root is not None and str(root) not in sys.path:
    sys.path.insert(0, str(root))

  try:
    importlib.import_module("mpc_osqp")
    parameters_module = importlib.import_module("MPC_Controller.Parameters")
    quadruped_module = importlib.import_module("MPC_Controller.common.Quadruped")
    runner_module = importlib.import_module(
      "MPC_Controller.robot_runner.RobotRunnerMin"
    )
  except (ImportError, SystemExit) as exc:
    root_hint = root or f"${CONTROLLER_ROOT_ENV}"
    raise RuntimeError(
      "RL-MPC native controller is unavailable for this Python interpreter. "
      f"Controller root: {root_hint}. Rebuild mpc_osqp with pybind11 >= 2.13 "
      "using scripts/build_rl_mpc_solver.py before creating the environment."
    ) from exc

  return (
    parameters_module.Parameters,
    quadruped_module.RobotType,
    runner_module.RobotRunnerMin,
  )


class LegacyG23MpcController:
  """One stateful G23 controller backed by rl-mpc-locomotion."""

  def __init__(self, cfg: LegacyMpcBackendCfg):
    if not np.isfinite(cfg.controller_dt) or cfg.controller_dt <= 0.0:
      raise ValueError("controller_dt must be finite and positive")
    if isinstance(cfg.iterations_between_mpc, bool) or not isinstance(
      cfg.iterations_between_mpc, int
    ):
      raise ValueError("iterations_between_mpc must be a positive integer")
    if cfg.iterations_between_mpc <= 0:
      raise ValueError("iterations_between_mpc must be a positive integer")

    stand_values = {
      "stand_enter_linear_speed": cfg.stand_enter_linear_speed,
      "stand_enter_yaw_rate": cfg.stand_enter_yaw_rate,
      "stand_exit_linear_speed": cfg.stand_exit_linear_speed,
      "stand_exit_yaw_rate": cfg.stand_exit_yaw_rate,
      "stand_command_hold_time": cfg.stand_command_hold_time,
      "stand_contact_hold_time": cfg.stand_contact_hold_time,
      "stand_contact_loss_time": cfg.stand_contact_loss_time,
      "stand_arm_timeout": cfg.stand_arm_timeout,
      "stand_max_linear_speed": cfg.stand_max_linear_speed,
      "stand_max_vertical_speed": cfg.stand_max_vertical_speed,
      "stand_max_yaw_rate": cfg.stand_max_yaw_rate,
      "stand_max_roll_pitch_rate": cfg.stand_max_roll_pitch_rate,
      "stand_foot_search_rate": cfg.stand_foot_search_rate,
      "stand_foot_search_depth": cfg.stand_foot_search_depth,
      "stand_mpc_xy_position_weight": cfg.stand_mpc_xy_position_weight,
      "stand_mpc_vxy_weight": cfg.stand_mpc_vxy_weight,
      "stand_hipx_kp": cfg.stand_hipx_kp,
      "stand_hipx_kd": cfg.stand_hipx_kd,
    }
    for name, value in stand_values.items():
      if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    for name, gains in (
      ("stand_kp_cartesian", cfg.stand_kp_cartesian),
      ("stand_kd_cartesian", cfg.stand_kd_cartesian),
    ):
      values = np.asarray(gains, dtype=np.float64)
      if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain three finite values")
      if np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if cfg.stand_exit_linear_speed <= cfg.stand_enter_linear_speed:
      raise ValueError("stand_exit_linear_speed must exceed stand_enter_linear_speed")
    if cfg.stand_exit_yaw_rate <= cfg.stand_enter_yaw_rate:
      raise ValueError("stand_exit_yaw_rate must exceed stand_enter_yaw_rate")

    parameters, robot_type, runner_cls = _load_external_modules(cfg)
    parameters.bridge_MPC_to_RL = True
    parameters.controller_dt = cfg.controller_dt
    parameters.flat_ground = cfg.flat_ground
    parameters.set_mpc_backend(cfg.formulation)

    if cfg.body_mass is not None and (
      not np.isfinite(cfg.body_mass) or cfg.body_mass <= 0.0
    ):
      raise ValueError("body_mass must be positive and finite")
    if cfg.foot_landing_height is not None and not np.isfinite(cfg.foot_landing_height):
      raise ValueError("foot_landing_height must be finite")

    self._runner = runner_cls()
    self._runner.init(
      robot_type.G23,
      body_mass=cfg.body_mass,
      foot_landing_height=cfg.foot_landing_height,
      iterations_between_mpc=cfg.iterations_between_mpc,
      gait_period=cfg.gait_period,
      enable_stand_mode=cfg.enable_stand_mode,
      stand_enter_linear_speed=cfg.stand_enter_linear_speed,
      stand_enter_yaw_rate=cfg.stand_enter_yaw_rate,
      stand_exit_linear_speed=cfg.stand_exit_linear_speed,
      stand_exit_yaw_rate=cfg.stand_exit_yaw_rate,
      stand_command_hold_time=cfg.stand_command_hold_time,
      stand_contact_hold_time=cfg.stand_contact_hold_time,
      stand_contact_loss_time=cfg.stand_contact_loss_time,
      stand_arm_timeout=cfg.stand_arm_timeout,
      stand_max_linear_speed=cfg.stand_max_linear_speed,
      stand_max_vertical_speed=cfg.stand_max_vertical_speed,
      stand_max_yaw_rate=cfg.stand_max_yaw_rate,
      stand_max_roll_pitch_rate=cfg.stand_max_roll_pitch_rate,
      stand_foot_search_rate=cfg.stand_foot_search_rate,
      stand_foot_search_depth=cfg.stand_foot_search_depth,
      stand_mpc_xy_position_weight=cfg.stand_mpc_xy_position_weight,
      stand_mpc_vxy_weight=cfg.stand_mpc_vxy_weight,
      stand_kp_cartesian=cfg.stand_kp_cartesian,
      stand_kd_cartesian=cfg.stand_kd_cartesian,
      stand_hipx_kp=cfg.stand_hipx_kp,
      stand_hipx_kd=cfg.stand_hipx_kd,
      symmetric_residuals=cfg.symmetric_residuals,
      stop_reposition_steps=cfg.stop_reposition_steps,
    )

  @property
  def gait_mode(self) -> int:
    mode = int(self._runner.cMPC.gait_mode)
    try:
      MpcGaitMode(mode)
    except ValueError as exc:
      raise RuntimeError(f"MPC controller returned invalid gait mode {mode}") from exc
    return mode

  @property
  def stand_ready(self) -> bool:
    return bool(self._runner.cMPC.stand_ready)

  @property
  def mpc_solver_failure_count(self) -> int:
    """Number of failed native force solves in the current episode."""
    return int(self._runner.cMPC.mpc_solver_failure_count)

  def reset(self) -> None:
    self._runner.reset()

  @property
  def observation_snapshot(self) -> np.ndarray:
    """Nominal XY (8), planned contact (4), applied foot feedforward (12)."""
    controller = self._runner.cMPC
    forces = np.concatenate(
      [
        command.forceFeedForward.reshape(3)
        for command in self._runner._legController.commands
      ]
    )
    return np.concatenate(
      (
        controller.foothold_heuristic,
        controller.policy_contact_state,
        forces,
      )
    ).astype(np.float32)

  def step(
    self,
    dof_states: np.ndarray,
    body_states: np.ndarray,
    command: np.ndarray,
    foot_placement_offsets: np.ndarray,
    foot_contacts: np.ndarray | None = None,
    terrain_height_samples: np.ndarray | None = None,
  ) -> np.ndarray:
    torque = self._runner.run(
      dof_states,
      body_states,
      command,
      foot_placement_offsets=foot_placement_offsets,
      foot_contacts=foot_contacts,
      terrain_height_samples=terrain_height_samples,
    )
    torque = np.asarray(torque, dtype=np.float32)
    if torque.shape != (12,):
      raise RuntimeError(
        f"MPC controller returned torque shape {torque.shape}, expected (12,)"
      )
    if not np.all(np.isfinite(torque)):
      raise RuntimeError("MPC controller returned non-finite torque")
    return torque


def make_legacy_g23_mpc_controller(cfg: LegacyMpcBackendCfg) -> MpcController:
  """Construct one controller instance without importing it at registry time."""
  return LegacyG23MpcController(cfg)
