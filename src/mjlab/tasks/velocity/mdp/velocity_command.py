from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _validate_directional_bucket_fractions(
  rel_forward_envs: float, rel_backward_envs: float
) -> None:
  for name, value in (
    ("rel_forward_envs", rel_forward_envs),
    ("rel_backward_envs", rel_backward_envs),
  ):
    if not 0.0 <= value <= 1.0:
      raise ValueError(f"{name} must be within [0, 1], got {value}.")
  if rel_forward_envs + rel_backward_envs > 1.0:
    raise ValueError(
      "rel_forward_envs + rel_backward_envs must not exceed 1, got "
      f"{rel_forward_envs + rel_backward_envs}."
    )


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")
    _validate_directional_bucket_fractions(
      self.cfg.rel_forward_envs, self.cfg.rel_backward_envs
    )

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)
    self.is_backward_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # 2026-09-02 stair-training update: keep per-episode, per-direction
    # tracking statistics for performance-gated command curricula.  These are
    # intentionally separate from ``metrics``: the curriculum consumes the
    # per-environment values before CommandTerm.reset() aggregates them.
    self._forward_tracking_sum = torch.zeros(self.num_envs, device=self.device)
    self._forward_tracking_count = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._backward_tracking_sum = torch.zeros(self.num_envs, device=self.device)
    self._backward_tracking_count = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

    # Populated only during interactive play. The Python set keeps the regular
    # training compute path free from device synchronization.
    self._manual_env_ids: set[int] = set()
    self._manual_env_ids_tensor: torch.Tensor | None = None
    self._manual_command_b = torch.zeros_like(self.vel_command_b)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def set_manual_command(
    self, env_idx: int, command: Sequence[float] | None
  ) -> tuple[float, float, float] | None:
    """Set a body-frame command for one env, or ``None`` for random mode.

    Call this on the simulation thread. ``play`` routes terminal input through the
    viewer action queue before invoking it.
    """
    if not 0 <= env_idx < self.num_envs:
      raise IndexError(
        f"Environment index {env_idx} is outside [0, {self.num_envs - 1}]."
      )
    if command is None:
      if env_idx not in self._manual_env_ids:
        return None
      self._manual_env_ids.remove(env_idx)
      self._refresh_manual_env_ids()
      env_ids = torch.tensor([env_idx], dtype=torch.long, device=self.device)
      self._resample(env_ids)
      self._update_command(env_ids)
      self._apply_manual_commands()
      self._disable_joystick_override()
      self._invalidate_observation_cache()
      return None
    if len(command) != 3:
      raise ValueError(
        f"Manual velocity command must have 3 values, got {len(command)}."
      )

    values = (float(command[0]), float(command[1]), float(command[2]))
    if not all(math.isfinite(value) for value in values):
      raise ValueError("Manual velocity command values must be finite.")
    ranges = (
      self.cfg.ranges.lin_vel_x,
      self.cfg.ranges.lin_vel_y,
      self.cfg.ranges.ang_vel_z,
    )
    clamped = (
      max(ranges[0][0], min(values[0], ranges[0][1])),
      max(ranges[1][0], min(values[1], ranges[1][1])),
      max(ranges[2][0], min(values[2], ranges[2][1])),
    )
    self._manual_command_b[env_idx] = torch.tensor(
      clamped, dtype=self.vel_command_b.dtype, device=self.device
    )
    self._manual_env_ids.add(env_idx)
    self._refresh_manual_env_ids()

    # Manual commands are body-frame setpoints and take precedence over every
    # random-command subtype.
    self.is_heading_env[env_idx] = False
    self.is_standing_env[env_idx] = False
    self.is_world_env[env_idx] = False
    self.is_forward_env[env_idx] = False
    self.is_backward_env[env_idx] = False
    self.vel_command_b[env_idx] = self._manual_command_b[env_idx]
    self.vel_command_w[env_idx] = self._manual_command_b[env_idx]
    self._disable_joystick_override()
    self._invalidate_observation_cache()
    return clamped

  def _disable_joystick_override(self) -> None:
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      self._joystick_enabled.value = False

  def _invalidate_observation_cache(self) -> None:
    observation_manager = getattr(self._env, "observation_manager", None)
    if observation_manager is not None:
      observation_manager.invalidate_cache()

  def _refresh_manual_env_ids(self) -> None:
    if self._manual_env_ids:
      self._manual_env_ids_tensor = torch.tensor(
        sorted(self._manual_env_ids), dtype=torch.long, device=self.device
      )
    else:
      self._manual_env_ids_tensor = None

  def _apply_manual_commands(self) -> None:
    env_ids = self._manual_env_ids_tensor
    if env_ids is None:
      return
    self.vel_command_b[env_ids] = self._manual_command_b[env_ids]
    self.vel_command_w[env_ids] = self._manual_command_b[env_ids]

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  # 2026-09-02 stair-training update: expose the completed episode's raw
  # directional accumulators without resetting them.  Curriculum terms run
  # before command reset, so callers can safely select the resetting envs.
  def get_directional_tracking_stats(
    self, env_ids: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    return {
      "forward_tracking_sum": self._forward_tracking_sum[env_ids],
      "forward_tracking_count": self._forward_tracking_count[env_ids],
      "backward_tracking_sum": self._backward_tracking_sum[env_ids],
      "backward_tracking_count": self._backward_tracking_count[env_ids],
    }

  # 2026-09-02 stair-training update: a resumed curriculum may restore a
  # later command stage after the environment's initial stage-0 reset.
  def resample_after_curriculum_restore(self) -> None:
    """Resample every command from the restored stage without changing robot state."""
    env_ids = torch.arange(self.num_envs, device=self.device)
    self._forward_tracking_sum.zero_()
    self._forward_tracking_count.zero_()
    self._backward_tracking_sum.zero_()
    self._backward_tracking_count.zero_()
    self._resample(env_ids)
    self._update_command(env_ids)
    self._apply_manual_commands()
    self._invalidate_observation_cache()

  def _accumulate_directional_tracking(self, dt: float | torch.Tensor) -> None:
    """Accumulate bounded planar tracking scores for positive-duration steps."""
    if isinstance(dt, torch.Tensor):
      active = dt.to(device=self.device) > 0.0
    else:
      if dt <= 0.0:
        return
      active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    planar_error_sq = torch.sum(
      torch.square(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2]
      ),
      dim=-1,
    )
    # Match the linear-velocity reward's 0.5 m/s kernel width while excluding
    # vertical velocity, which is expected during stair traversal.
    tracking_score = torch.exp(-planar_error_sq / 0.5**2)

    forward = self.is_forward_env & active
    backward = self.is_backward_env & active
    self._forward_tracking_sum += tracking_score * forward
    self._forward_tracking_count += forward
    self._backward_tracking_sum += tracking_score * backward
    self._backward_tracking_count += backward

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    # Draw mutually exclusive straight-forward and straight-backward buckets.
    # Standing commands are independent and take precedence, so exclude them
    # from the directional masks and let _update_command() zero them as usual.
    selector = r.uniform_(0.0, 1.0)
    is_non_standing = ~self.is_standing_env[env_ids]
    is_forward = (selector < self.cfg.rel_forward_envs) & is_non_standing
    is_backward = (
      (selector >= self.cfg.rel_forward_envs)
      & (selector < self.cfg.rel_forward_envs + self.cfg.rel_backward_envs)
      & is_non_standing
    )
    self.is_forward_env[env_ids] = is_forward
    self.is_backward_env[env_ids] = is_backward

    # Dedicated directional buckets are body-frame straight commands. Prevent
    # later heading control or world-frame rotation from changing their yaw or
    # lateral velocity.
    directional_ids = env_ids[is_forward | is_backward]
    self.is_heading_env[directional_ids] = False
    self.is_world_env[directional_ids] = False

    fwd_ids = env_ids[is_forward]
    if len(fwd_ids) > 0:
      x_lower, x_upper = self.cfg.ranges.lin_vel_x
      # A two-sided range describes distinct forward/backward safety
      # envelopes.  Do not let abs() mirror the larger side across zero (for
      # example, (-0.3, 0.5) must never create a -0.5 command).  One-sided
      # ranges retain their historical meaning as a speed-magnitude range.
      forward_max = (
        x_upper if x_lower < 0.0 < x_upper else max(abs(x_lower), abs(x_upper))
      )
      forward_min = min(0.3, forward_max)
      self.vel_command_b[fwd_ids, 0] = (
        self.vel_command_b[fwd_ids, 0].abs().clamp(min=forward_min, max=forward_max)
      )
      self.vel_command_b[fwd_ids, 1] = 0.0
      self.vel_command_b[fwd_ids, 2] = 0.0

    bwd_ids = env_ids[is_backward]
    if len(bwd_ids) > 0:
      x_lower, x_upper = self.cfg.ranges.lin_vel_x
      backward_max = (
        -x_lower if x_lower < 0.0 < x_upper else max(abs(x_lower), abs(x_upper))
      )
      backward_min = min(0.3, backward_max)
      self.vel_command_b[bwd_ids, 0] = (
        -self.vel_command_b[bwd_ids, 0].abs().clamp(min=backward_min, max=backward_max)
      )
      self.vel_command_b[bwd_ids, 1] = 0.0
      self.vel_command_b[bwd_ids, 2] = 0.0

    self.vel_command_w[directional_ids] = self.vel_command_b[directional_ids]

    if self.cfg.sample_single_axis_commands:
      # Some residual-controller tasks deliberately begin from a pure-MPC
      # envelope that has been certified one axis at a time. Keep their random
      # training samples inside that tested envelope; manual play commands are
      # intentionally unaffected so mixed-command fine-tuning remains possible.
      is_general = is_non_standing & ~is_forward & ~is_backward
      general_ids = env_ids[is_general]
      if len(general_ids) > 0:
        selected_axes = torch.randint(0, 3, (len(general_ids),), device=self.device)
        keep = torch.nn.functional.one_hot(selected_axes, num_classes=3).bool()
        self.vel_command_b[general_ids] *= keep
        self.vel_command_w[general_ids] = self.vel_command_b[general_ids]
        self.is_heading_env[general_ids] = False
        self.is_world_env[general_ids] = False

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    # 2026-09-02 stair-training update: reset only the requested environments;
    # asynchronous vector-environment episodes must retain their own statistics.
    assert isinstance(env_ids, torch.Tensor)
    self._forward_tracking_sum[env_ids] = 0.0
    self._forward_tracking_count[env_ids] = 0
    self._backward_tracking_sum[env_ids] = 0.0
    self._backward_tracking_count[env_ids] = 0
    self._apply_manual_commands()
    if self.cfg.init_velocity_prob > 0.0:
      assert isinstance(env_ids, torch.Tensor)
      r = torch.empty(len(env_ids), device=self.device)
      init_ids = env_ids[r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob]
      if len(init_ids) > 0:
        # Start these envs already moving at the commanded planar velocity.
        # Safe pre-forward: the body-frame write reads orientation from qpos.
        vel_b = torch.zeros(len(init_ids), 6, device=self.device)
        vel_b[:, :2] = self.vel_command_b[init_ids, :2]
        vel_b[:, 5] = self.vel_command_b[init_ids, 2]
        self.robot.write_root_link_velocity_b_to_sim(vel_b, env_ids=init_ids)
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # Pure function of the current state; refreshing all envs is safe.
    del env_ids
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      heading_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[heading_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[heading_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    # 2026-09-02 stair-training update: measure the command that governed the
    # just-finished step before timer expiry can resample it.  A per-env zero dt
    # on auto-reset prevents freshly reset environments from adding a fake sample.
    self._accumulate_directional_tracking(dt)
    super().compute(dt, env_ids)
    self._apply_manual_commands()
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0
  """Fraction of environments that use world-frame velocity commands.
  World-frame envs sample linear velocity in world frame and rotate to body
  frame each step, so the command direction stays fixed in the world."""
  rel_forward_envs: float = 0.0
  """Fraction of non-standing environments that receive straight-forward
  body-frame commands (positive lin_vel_x, zero lin_vel_y and ang_vel_z).
  Forward and backward buckets are mutually exclusive."""
  rel_backward_envs: float = 0.0
  """Fraction of non-standing environments that receive straight-backward
  body-frame commands (negative lin_vel_x, zero lin_vel_y and ang_vel_z).
  The remaining ``1 - rel_forward_envs - rel_backward_envs`` environments use
  general sampled commands."""
  init_velocity_prob: float = 0.0
  """Probability that an env starts its episode already moving at its sampled
  planar command velocity. Applied on reset only."""
  sample_single_axis_commands: bool = False
  """Sample at most one of ``vx``, ``vy`` and ``wz`` in general-command envs.

  Straight forward/backward buckets already use only ``vx``. Manual commands
  remain unrestricted within ``ranges`` so a trained policy can later be
  evaluated or fine-tuned with mixed commands.
  """

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    _validate_directional_bucket_fractions(
      self.rel_forward_envs, self.rel_backward_envs
    )
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
