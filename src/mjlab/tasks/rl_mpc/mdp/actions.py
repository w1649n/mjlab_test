"""RL-MPC action term for planar G23 foot-placement residuals."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.tasks.rl_mpc.controller.backend import (
  LegacyMpcBackendCfg,
  MpcController,
  MpcGaitMode,
  make_legacy_g23_mpc_controller,
)
from mjlab.tasks.rl_mpc.controller.foot_placement import (
  FOOT_PLACEMENT_ACTION_DIM,
  FootPlacementOffsetCfg,
  FootPlacementOffsetLatch,
)
from mjlab.tasks.rl_mpc.controller.observation_history import MpcObservationHistory
from mjlab.tasks.rl_mpc.controller.state_adapter import (
  G23_MPC_JOINT_ORDER,
  pack_mpc_controller_batch,
)
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_G23_TORQUE_LIMITS = (24.0, 24.0, 36.0) * 4
_G23_FOOT_CONTACT_ORDER = (
  "FL_foot_collision",
  "FR_foot_collision",
  "HL_foot_collision",
  "HR_foot_collision",
)


@dataclass(frozen=True)
class TerrainHeightSampleBatch:
  """Validated local terrain samples and per-environment miss diagnostics."""

  samples: torch.Tensor
  miss_fraction: torch.Tensor
  miss_count: torch.Tensor
  below_min_valid_fraction: torch.Tensor
  total_miss_count: int
  affected_envs: tuple[int, ...]


def extract_terrain_height_samples(
  *,
  hit_pos_w: torch.Tensor,
  distances: torch.Tensor,
  env_origins: torch.Tensor,
  min_valid_fraction: float,
) -> TerrainHeightSampleBatch:
  """Validate ray hits and repair isolated misses with same-env nearest Z.

  A missed ray's hit position is its ray origin, so its XY coordinate remains
  the correct query location. Only Z is replaced, using the closest valid ray
  in that environment. Environments with no valid ray receive a finite zero Z;
  callers must reject or skip them via ``below_min_valid_fraction``.
  """
  if (
    isinstance(min_valid_fraction, bool)
    or not math.isfinite(min_valid_fraction)
    or not 0.0 < min_valid_fraction <= 1.0
  ):
    raise ValueError(
      f"min_valid_fraction must be finite and in (0, 1], got {min_valid_fraction}"
    )
  if hit_pos_w.ndim != 3 or hit_pos_w.shape[-1] != 3:
    raise ValueError(
      f"hit_pos_w must have shape (num_envs, num_rays, 3), got {tuple(hit_pos_w.shape)}"
    )
  num_envs, num_rays, _ = hit_pos_w.shape
  if num_envs == 0 or num_rays == 0:
    raise ValueError("terrain scan must contain at least one environment and ray")
  if distances.shape != (num_envs, num_rays):
    raise ValueError(
      f"distances must have shape ({num_envs}, {num_rays}), "
      f"got {tuple(distances.shape)}"
    )
  if env_origins.shape != (num_envs, 3):
    raise ValueError(
      f"env_origins must have shape ({num_envs}, 3), got {tuple(env_origins.shape)}"
    )
  tensors = {
    "hit_pos_w": hit_pos_w,
    "distances": distances,
    "env_origins": env_origins,
  }
  for name, value in tensors.items():
    if not value.is_floating_point():
      raise TypeError(f"{name} must be a floating-point tensor")
    if value.device != hit_pos_w.device:
      raise ValueError("terrain scan tensors must be on the same device")

  missed = distances < 0.0
  miss_count = torch.count_nonzero(missed, dim=1)
  finite_status = torch.stack(
    [torch.all(torch.isfinite(value)) for value in tensors.values()]
  )
  # One device-to-host synchronization covers both mandatory finite validation
  # and the miss fast path.
  scan_status = torch.cat(
    (finite_status.to(dtype=torch.long), miss_count.sum().reshape(1))
  ).tolist()
  for name, is_finite in zip(tensors, scan_status[:-1], strict=True):
    if not is_finite:
      raise ValueError(f"{name} contains NaN or Inf")
  total_miss_count = int(scan_status[-1])
  miss_fraction = miss_count.to(dtype=torch.float32) / float(num_rays)
  minimum_valid_rays = math.ceil(min_valid_fraction * num_rays)
  below_min_valid_fraction = (num_rays - miss_count) < minimum_valid_rays
  samples = (hit_pos_w - env_origins[:, None, :]).clone()

  if total_miss_count == 0:
    return TerrainHeightSampleBatch(
      samples=samples,
      miss_fraction=miss_fraction,
      miss_count=miss_count,
      below_min_valid_fraction=below_min_valid_fraction,
      total_miss_count=0,
      affected_envs=(),
    )

  affected_envs = tuple(torch.nonzero(miss_count, as_tuple=False).flatten().tolist())
  for env_idx in affected_envs:
    missed_indices = torch.nonzero(missed[env_idx], as_tuple=False).flatten()
    valid_indices = torch.nonzero(~missed[env_idx], as_tuple=False).flatten()
    if valid_indices.numel() == 0:
      samples[env_idx, missed_indices, 2] = 0.0
      continue
    miss_xy = samples[env_idx, missed_indices, :2]
    valid_xy = samples[env_idx, valid_indices, :2]
    nearest_valid = torch.cdist(miss_xy, valid_xy).argmin(dim=1)
    samples[env_idx, missed_indices, 2] = samples[
      env_idx, valid_indices[nearest_valid], 2
    ]

  return TerrainHeightSampleBatch(
    samples=samples,
    miss_fraction=miss_fraction,
    miss_count=miss_count,
    below_min_valid_fraction=below_min_valid_fraction,
    total_miss_count=total_miss_count,
    affected_envs=affected_envs,
  )


@dataclass(kw_only=True)
class MpcFootPlacementActionCfg(ActionTermCfg):
  """Configure an eight-dimensional residual foothold action."""

  command_name: str = "twist"
  controller_root: str | None = None
  formulation: Literal["native", "syncai"] = "syncai"
  controller_dt: float = 0.01
  iterations_between_mpc: int = 2
  gait_period: float | None = None
  flat_ground: bool = False
  body_mass: float | None = None
  foot_landing_height: float | None = None
  contact_sensor_name: str = "feet_ground_contact"
  terrain_sensor_name: str | None = None
  terrain_scan_miss_policy: Literal["error", "nearest_valid"] = "error"
  terrain_scan_min_valid_fraction: float = 0.9
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
  foot_offset_scale_xy: tuple[float, float] = (0.08, 0.08)
  joint_names: tuple[str, ...] = G23_MPC_JOINT_ORDER
  torque_limits: tuple[float, ...] = _G23_TORQUE_LIMITS
  fail_on_controller_error: bool = True
  controller_factory: Callable[[], MpcController] | None = None

  def build(self, env: ManagerBasedRlEnv) -> MpcFootPlacementAction:
    return MpcFootPlacementAction(self, env)


class MpcFootPlacementAction(ActionTerm):
  """Run one stateful MPC controller per environment and apply its torques."""

  cfg: MpcFootPlacementActionCfg
  _entity: Entity

  def __init__(self, cfg: MpcFootPlacementActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    controller_substeps = cfg.controller_dt / env.physics_dt
    policy_ticks = env.step_dt / cfg.controller_dt
    if (
      controller_substeps < 1
      or policy_ticks < 1
      or not math.isclose(controller_substeps, round(controller_substeps))
      or not math.isclose(policy_ticks, round(policy_ticks))
    ):
      raise ValueError(
        "RL-MPC physics, controller and policy periods must be integer multiples"
      )
    self._controller_substeps = round(controller_substeps)
    self._physics_since_action = 0
    self._action_history = torch.zeros((self.num_envs, 4, 8), device=self.device)
    self._observation_history = MpcObservationHistory(self.num_envs, self.device)
    self._mpc_snapshot = torch.zeros((self.num_envs, 24), device=self.device)
    self._mpc_snapshot[:, 8:12] = 1.0
    if cfg.terrain_scan_miss_policy not in {"error", "nearest_valid"}:
      raise ValueError(
        "terrain_scan_miss_policy must be 'error' or 'nearest_valid', "
        f"got {cfg.terrain_scan_miss_policy!r}"
      )
    if (
      isinstance(cfg.terrain_scan_min_valid_fraction, bool)
      or not math.isfinite(cfg.terrain_scan_min_valid_fraction)
      or not 0.0 < cfg.terrain_scan_min_valid_fraction <= 1.0
    ):
      raise ValueError(
        "terrain_scan_min_valid_fraction must be finite and in (0, 1], "
        f"got {cfg.terrain_scan_min_valid_fraction}"
      )

    joint_ids, joint_names = self._entity.find_joints(
      cfg.joint_names, preserve_order=True
    )
    if tuple(joint_names) != tuple(cfg.joint_names):
      raise ValueError(
        f"G23 MPC joint order mismatch: expected {cfg.joint_names}, got {tuple(joint_names)}"
      )
    if len(cfg.torque_limits) != len(joint_ids):
      raise ValueError("torque_limits must contain one value per MPC joint")

    contact_sensor = self._env.scene[cfg.contact_sensor_name]
    if not isinstance(contact_sensor, ContactSensor):
      raise TypeError(f"{cfg.contact_sensor_name!r} must refer to a ContactSensor")
    if tuple(contact_sensor.primary_names) != _G23_FOOT_CONTACT_ORDER:
      raise ValueError(
        "G23 foot contact order mismatch: expected "
        f"{_G23_FOOT_CONTACT_ORDER}, got {tuple(contact_sensor.primary_names)}"
      )
    if contact_sensor.data.found is None:
      raise ValueError(
        f"Contact sensor {cfg.contact_sensor_name!r} must provide the 'found' field"
      )
    self._contact_sensor = contact_sensor

    self._terrain_sensor: RayCastSensor | None = None
    if cfg.terrain_sensor_name is not None:
      terrain_sensor = self._env.scene[cfg.terrain_sensor_name]
      if not isinstance(terrain_sensor, RayCastSensor):
        raise TypeError(f"{cfg.terrain_sensor_name!r} must refer to a RayCastSensor")
      if terrain_sensor.num_frames != 1:
        raise ValueError(
          f"{cfg.terrain_sensor_name!r} must expose one base-frame terrain map"
        )
      self._terrain_sensor = terrain_sensor

    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._torque_limits = torch.tensor(
      cfg.torque_limits, device=self.device, dtype=torch.float32
    ).view(1, -1)
    self._offsets = FootPlacementOffsetLatch(
      self.num_envs,
      self.device,
      FootPlacementOffsetCfg(scale_xy=cfg.foot_offset_scale_xy),
    )
    self._torques = torch.zeros(
      self.num_envs, len(joint_ids), device=self.device, dtype=torch.float32
    )
    self.controller_error_count = 0
    self._terrain_scan_miss_fraction = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    self._terrain_scan_failed = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._terrain_scan_initialized = False
    self._terrain_scan_missed_ray_count = 0
    self._terrain_scan_failure_count = 0

    if cfg.controller_factory is None:
      backend_cfg = LegacyMpcBackendCfg(
        controller_root=cfg.controller_root,
        formulation=cfg.formulation,
        controller_dt=cfg.controller_dt,
        iterations_between_mpc=cfg.iterations_between_mpc,
        gait_period=cfg.gait_period,
        flat_ground=cfg.flat_ground,
        body_mass=cfg.body_mass,
        foot_landing_height=cfg.foot_landing_height,
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
      self._controllers = [
        make_legacy_g23_mpc_controller(backend_cfg) for _ in range(self.num_envs)
      ]
    else:
      self._controllers = [cfg.controller_factory() for _ in range(self.num_envs)]
    self._gait_modes = torch.tensor(
      [int(controller.gait_mode) for controller in self._controllers],
      device=self.device,
      dtype=torch.long,
    )
    self._stand_ready = torch.tensor(
      [bool(controller.stand_ready) for controller in self._controllers],
      device=self.device,
      dtype=torch.bool,
    )

  @property
  def action_dim(self) -> int:
    return FOOT_PLACEMENT_ACTION_DIM

  @property
  def raw_action(self) -> torch.Tensor:
    return self._offsets.raw_actions

  @property
  def processed_offsets(self) -> torch.Tensor:
    """Latest bounded yaw-frame offsets in metres, shaped ``(N, 4, 2)``."""
    return self._offsets.pending_offsets

  @property
  def torques(self) -> torch.Tensor:
    return self._torques

  @property
  def gait_modes(self) -> torch.Tensor:
    """Actual controller mode for each environment (TROT/STOPPING/STAND)."""
    return self._gait_modes

  @property
  def stand_ready(self) -> torch.Tensor:
    """Whether each controller has confirmed continuous four-foot contact."""
    return self._stand_ready

  @property
  def terrain_scan_miss_fraction(self) -> torch.Tensor:
    """Fraction of missed rays in the latest scan for each environment."""
    return self._terrain_scan_miss_fraction

  @property
  def terrain_scan_failed(self) -> torch.Tensor:
    """Environments whose latest scan was too incomplete for controller use."""
    return self._terrain_scan_failed

  @property
  def terrain_scan_missed_ray_count(self) -> int:
    """Cumulative number of missed terrain rays since action construction."""
    return self._terrain_scan_missed_ray_count

  @property
  def terrain_scan_failure_count(self) -> int:
    """Cumulative number of environment scans rejected or skipped."""
    return self._terrain_scan_failure_count

  @property
  def mpc_solver_failure_count(self) -> int:
    """Aggregate native MPC solve failures across all environments."""
    return sum(
      int(getattr(controller, "mpc_solver_failure_count", 0))
      for controller in self._controllers
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    self._offsets.process_actions(actions)
    self._action_history[:, 1:] = self._action_history[:, :-1].clone()
    self._action_history[:, 0] = actions.clamp(-1.0, 1.0)
    self._physics_since_action = 0
    self._update_controller()

  @property
  def action_history(self) -> torch.Tensor:
    """Four clipped policy actions, newest first; zero padded after reset."""
    return self._action_history.flatten(start_dim=1)

  @property
  def state_history(self) -> torch.Tensor:
    """Five complete 75D frames, newest first, with physical last offsets."""
    data = self._entity.data
    rpy = torch.stack(euler_xyz_from_quat(data.root_link_quat_w), dim=-1)
    rpy = (rpy + math.pi) % (2 * math.pi) - math.pi
    command = self._env.command_manager.get_command(self.cfg.command_name)
    if command is None:
      raise RuntimeError("MPC velocity command is unavailable")
    command_world = command[:, :3].clone()
    c, s = rpy[:, 2].cos(), rpy[:, 2].sin()
    command_world[:, 0] = c * command[:, 0] - s * command[:, 1]
    command_world[:, 1] = s * command[:, 0] + c * command[:, 1]
    found = self._contact_sensor.data.found
    if found is None:
      raise RuntimeError("MPC foot contact data is unavailable")
    actual_contact = found.reshape(self.num_envs, 4, -1).gt(0).any(dim=-1)
    mode = torch.nn.functional.one_hot(self._gait_modes, num_classes=3)
    frame = torch.cat(
      (
        rpy,
        data.root_link_lin_vel_w,
        data.root_link_ang_vel_w,
        data.joint_pos[:, self._joint_ids],
        data.joint_vel[:, self._joint_ids],
        self._mpc_snapshot[:, :8],
        actual_contact,
        self._mpc_snapshot[:, 8:12],
        self._mpc_snapshot[:, 12:] / 12.0,
        command_world,
        self.processed_offsets.flatten(start_dim=1),
        mode,
      ),
      dim=1,
    )
    return self._observation_history.update(frame, self._env.common_step_counter)

  def _update_controller(self) -> None:
    self._terrain_scan_failed.zero_()
    self._terrain_scan_miss_fraction.zero_()
    pending_offsets = self._offsets.pending_offsets
    command = self._env.command_manager.get_command(self.cfg.command_name)
    if command is None:
      raise RuntimeError(f"Command {self.cfg.command_name!r} is unavailable")

    found = self._contact_sensor.data.found
    if found is None:
      raise RuntimeError(
        f"Contact sensor {self.cfg.contact_sensor_name!r} has no 'found' data"
      )
    if found.shape[0] != self.num_envs or found.shape[1] % 4 != 0:
      raise RuntimeError(
        f"Foot contact data must have shape ({self.num_envs}, 4 * slots), "
        f"got {tuple(found.shape)}"
      )
    foot_contacts = found.reshape(self.num_envs, 4, -1).gt(0).any(dim=-1)

    terrain_height_samples = None
    failed_envs: set[int] = set()
    if self._terrain_sensor is not None:
      terrain_data = self._terrain_sensor.data
      terrain_batch = extract_terrain_height_samples(
        hit_pos_w=terrain_data.hit_pos_w,
        distances=terrain_data.distances,
        env_origins=self._env.scene.env_origins,
        min_valid_fraction=self.cfg.terrain_scan_min_valid_fraction,
      )
      self._terrain_scan_miss_fraction.copy_(terrain_batch.miss_fraction)
      missed_ray_count = terrain_batch.total_miss_count
      self._terrain_scan_missed_ray_count += missed_ray_count
      if self.cfg.terrain_scan_miss_policy == "error" and missed_ray_count:
        self._terrain_scan_failure_count += len(terrain_batch.affected_envs)
        raise RuntimeError(
          f"Terrain sensor {self.cfg.terrain_sensor_name!r} missed "
          f"{missed_ray_count} ground ray(s) in env(s) "
          f"{list(terrain_batch.affected_envs)}"
        )
      severe = terrain_batch.below_min_valid_fraction
      if missed_ray_count and bool(torch.any(severe).item()):
        affected = tuple(torch.nonzero(severe, as_tuple=False).flatten().tolist())
        severe_count = len(affected)
        self._terrain_scan_failure_count += severe_count
        # One established environment can recover through its per-env
        # termination below. A severe first scan or simultaneous severe scans
        # instead indicate a systemic sensor/terrain configuration failure.
        if not self._terrain_scan_initialized:
          raise RuntimeError(
            "Terrain scan is severely incomplete during initialization for "
            f"env(s) {list(affected)}"
          )
        if severe_count > 1:
          raise RuntimeError(
            "Terrain scan is severely incomplete in multiple environments: "
            f"{list(affected)}"
          )
        self._terrain_scan_failed.copy_(severe)
        failed_envs.update(affected)
      else:
        self._terrain_scan_initialized = True
      terrain_height_samples = terrain_batch.samples

    batch = pack_mpc_controller_batch(
      joint_pos=self._entity.data.joint_pos[:, self._joint_ids],
      joint_vel=self._entity.data.joint_vel[:, self._joint_ids],
      root_pos_w=self._entity.data.root_link_pos_w,
      root_quat_wxyz=self._entity.data.root_link_quat_w,
      root_lin_vel_w=self._entity.data.root_link_lin_vel_w,
      root_ang_vel_w=self._entity.data.root_link_ang_vel_w,
      env_origins=self._env.scene.env_origins,
      commands=command,
      foot_placement_offsets=pending_offsets,
      foot_contacts=foot_contacts,
      terrain_height_samples=terrain_height_samples,
    )

    torque_rows: list[np.ndarray] = []
    gait_modes: list[int] = []
    stand_ready: list[bool] = []
    for env_idx, controller in enumerate(self._controllers):
      if env_idx in failed_envs:
        torque_rows.append(np.zeros(len(self._joint_ids), dtype=np.float32))
        gait_modes.append(int(self._gait_modes[env_idx].item()))
        stand_ready.append(bool(self._stand_ready[env_idx].item()))
        continue
      try:
        if batch.terrain_height_samples is None:
          torque = controller.step(
            batch.dof_states[env_idx],
            batch.body_states[env_idx],
            batch.commands[env_idx],
            batch.foot_placement_offsets[env_idx],
            batch.foot_contacts[env_idx],
          )
        else:
          torque = controller.step(
            batch.dof_states[env_idx],
            batch.body_states[env_idx],
            batch.commands[env_idx],
            batch.foot_placement_offsets[env_idx],
            batch.foot_contacts[env_idx],
            batch.terrain_height_samples[env_idx],
          )
        gait_mode = int(controller.gait_mode)
        if gait_mode not in {int(mode) for mode in MpcGaitMode}:
          raise RuntimeError(f"MPC controller returned invalid gait mode {gait_mode}")
        is_stand_ready = bool(controller.stand_ready)
        snapshot = np.asarray(controller.observation_snapshot, dtype=np.float32)
        if snapshot.shape != (24,) or not np.isfinite(snapshot).all():
          raise RuntimeError("Invalid MPC observation snapshot")
        self._mpc_snapshot[env_idx] = torch.from_numpy(snapshot).to(self.device)
      except Exception as exc:
        self.controller_error_count += 1
        if self.cfg.fail_on_controller_error:
          raise RuntimeError(f"MPC controller failed for env {env_idx}") from exc
        torque = np.zeros(len(self._joint_ids), dtype=np.float32)
        gait_mode = int(self._gait_modes[env_idx].item())
        is_stand_ready = bool(self._stand_ready[env_idx].item())
      torque_rows.append(np.asarray(torque, dtype=np.float32))
      gait_modes.append(gait_mode)
      stand_ready.append(is_stand_ready)

    torques = torch.from_numpy(np.stack(torque_rows)).to(self.device)
    self._torques.copy_(torch.clamp(torques, -self._torque_limits, self._torque_limits))
    self._gait_modes.copy_(torch.tensor(gait_modes, device=self.device))
    self._stand_ready.copy_(torch.tensor(stand_ready, device=self.device))

  def apply_actions(self) -> None:
    if (
      self._physics_since_action > 0
      and self._physics_since_action % self._controller_substeps == 0
    ):
      self._update_controller()
    self._entity.set_joint_effort_target(self._torques, joint_ids=self._joint_ids)
    self._physics_since_action += 1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._offsets.reset(env_ids)
    self._action_history[env_ids] = 0.0
    self._observation_history.reset(env_ids)
    self._mpc_snapshot[env_ids] = 0.0
    self._mpc_snapshot[env_ids, 8:12] = 1.0
    self._torques[env_ids] = 0.0
    self._terrain_scan_miss_fraction[env_ids] = 0.0
    self._terrain_scan_failed[env_ids] = False

    if isinstance(env_ids, slice):
      selected = range(self.num_envs)[env_ids]
    else:
      selected = env_ids.detach().cpu().tolist()
    for env_idx in selected:
      self._controllers[env_idx].reset()
      self._gait_modes[env_idx] = int(self._controllers[env_idx].gait_mode)
      self._stand_ready[env_idx] = bool(self._controllers[env_idx].stand_ready)
