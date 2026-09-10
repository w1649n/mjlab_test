"""Map policy actions to planar residual foothold offsets.

The deployed policy output is leg-major and expressed in the controller's
unrotated XY coordinates::

  [FL_dx, FL_dy, FR_dx, FR_dy, HL_dx, HL_dy, HR_dx, HR_dy]

Offsets are latched independently when each leg enters swing.  Keeping the target
fixed for the remainder of the swing avoids moving the landing point whenever a
new policy action is produced.  The historical ``yaw_aligned_to_world`` helper
remains available for other callers, but this task does not use it: its external
controller executes ``Pf[x/y] += offset`` directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

LEG_ORDER = ("FL", "FR", "HL", "HR")
"""Leg order shared with the G23 MPC controller."""

FOOT_PLACEMENT_ACTION_NAMES = tuple(
  f"{leg}_d{axis}" for leg in LEG_ORDER for axis in ("x", "y")
)
"""Names of the eight leg-major policy outputs."""

FOOT_PLACEMENT_ACTION_DIM = len(FOOT_PLACEMENT_ACTION_NAMES)
FOOT_PLACEMENT_HARD_LIMIT_M = 0.08
"""Absolute safety limit for each planar foothold residual, in metres."""


@dataclass(frozen=True, kw_only=True)
class FootPlacementOffsetCfg:
  """Configuration for the foot-placement residual mapper.

  ``scale_xy`` converts a normalized policy action into metres.  It may be tuned,
  but the mapped result is always clipped to ``+/- 0.08 m`` on each axis.
  """

  scale_xy: tuple[float, float] = (0.08, 0.08)

  def __post_init__(self) -> None:
    if len(self.scale_xy) != 2:
      raise ValueError("scale_xy must contain exactly (x, y)")
    if not all(math.isfinite(value) and value >= 0.0 for value in self.scale_xy):
      raise ValueError("scale_xy values must be finite and non-negative")


def yaw_aligned_to_world(offsets: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
  """Rotate yaw-aligned planar offsets into the world frame.

  Args:
    offsets: Tensor shaped ``(num_envs, 4, 2)`` in yaw-aligned coordinates.
    base_yaw: Tensor shaped ``(num_envs,)`` containing world-frame base yaw.

  Returns:
    Tensor shaped ``(num_envs, 4, 2)`` in world-frame coordinates.
  """
  if offsets.ndim != 3 or offsets.shape[1:] != (len(LEG_ORDER), 2):
    raise ValueError(
      f"offsets must have shape (num_envs, {len(LEG_ORDER)}, 2); "
      f"got {tuple(offsets.shape)}"
    )
  if base_yaw.shape != (offsets.shape[0],):
    raise ValueError(
      f"base_yaw must have shape ({offsets.shape[0]},); got {tuple(base_yaw.shape)}"
    )
  if base_yaw.device != offsets.device:
    raise ValueError("base_yaw and offsets must be on the same device")

  cos_yaw = torch.cos(base_yaw).unsqueeze(-1)
  sin_yaw = torch.sin(base_yaw).unsqueeze(-1)
  dx = offsets[..., 0]
  dy = offsets[..., 1]
  return torch.stack((cos_yaw * dx - sin_yaw * dy, sin_yaw * dx + cos_yaw * dy), dim=-1)


class FootPlacementOffsetLatch:
  """Scale, bound, and latch residual foothold actions for vectorized environments."""

  def __init__(
    self,
    num_envs: int,
    device: torch.device | str,
    cfg: FootPlacementOffsetCfg | None = None,
    dtype: torch.dtype = torch.float32,
  ) -> None:
    if num_envs <= 0:
      raise ValueError("num_envs must be positive")

    self.num_envs = num_envs
    self.device = torch.device(device)
    self.dtype = dtype
    self.cfg = cfg if cfg is not None else FootPlacementOffsetCfg()

    self._scale_xy = torch.tensor(
      self.cfg.scale_xy, device=self.device, dtype=self.dtype
    ).view(1, 1, 2)
    self._raw_actions = torch.zeros(
      num_envs, FOOT_PLACEMENT_ACTION_DIM, device=self.device, dtype=self.dtype
    )
    self._pending_offsets = torch.zeros(
      num_envs, len(LEG_ORDER), 2, device=self.device, dtype=self.dtype
    )
    self._latched_offsets = torch.zeros_like(self._pending_offsets)
    self._was_swinging = torch.zeros(
      num_envs, len(LEG_ORDER), device=self.device, dtype=torch.bool
    )

  @property
  def raw_actions(self) -> torch.Tensor:
    """Most recent unscaled policy actions, shaped ``(num_envs, 8)``."""
    return self._raw_actions

  @property
  def pending_offsets(self) -> torch.Tensor:
    """Most recent mapped offsets awaiting the next swing entry."""
    return self._pending_offsets

  @property
  def latched_offsets(self) -> torch.Tensor:
    """Per-leg offsets fixed at the most recent swing entry."""
    return self._latched_offsets

  def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
    """Convert normalized eight-dimensional actions into bounded metre offsets."""
    expected_shape = (self.num_envs, FOOT_PLACEMENT_ACTION_DIM)
    if actions.shape != expected_shape:
      raise ValueError(
        f"actions must have shape {expected_shape}; got {tuple(actions.shape)}"
      )
    if actions.device != self.device:
      raise ValueError("actions must be on the latch device")
    if not torch.is_floating_point(actions):
      raise TypeError("actions must be a floating-point tensor")
    if not torch.all(torch.isfinite(actions)):
      raise ValueError("actions must contain only finite values")

    self._raw_actions.copy_(actions.to(dtype=self.dtype))
    normalized = torch.clamp(self._raw_actions, min=-1.0, max=1.0)
    mapped = normalized.view(self.num_envs, len(LEG_ORDER), 2) * self._scale_xy
    self._pending_offsets.copy_(
      torch.clamp(
        mapped,
        min=-FOOT_PLACEMENT_HARD_LIMIT_M,
        max=FOOT_PLACEMENT_HARD_LIMIT_M,
      )
    )
    return self._pending_offsets

  def latch_on_swing_entry(self, swing_state: torch.Tensor) -> torch.Tensor:
    """Latch the latest offset independently on each stance-to-swing transition.

    ``swing_state`` may be boolean or a continuous gait phase.  Positive values
    are treated as swing.  Repeated calls during the same swing leave that leg's
    landing offset unchanged.
    """
    expected_shape = (self.num_envs, len(LEG_ORDER))
    if swing_state.shape != expected_shape:
      raise ValueError(
        f"swing_state must have shape {expected_shape}; got {tuple(swing_state.shape)}"
      )
    if swing_state.device != self.device:
      raise ValueError("swing_state must be on the latch device")
    if torch.is_floating_point(swing_state) and not torch.all(
      torch.isfinite(swing_state)
    ):
      raise ValueError("swing_state must contain only finite values")

    is_swinging = swing_state if swing_state.dtype == torch.bool else swing_state > 0
    entering_swing = is_swinging & ~self._was_swinging
    self._latched_offsets[entering_swing] = self._pending_offsets[entering_swing]
    self._was_swinging.copy_(is_swinging)
    return self._latched_offsets

  def latched_offsets_world(self, base_yaw: torch.Tensor) -> torch.Tensor:
    """Return the latched residuals rotated into world coordinates."""
    return yaw_aligned_to_world(self._latched_offsets, base_yaw)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear raw, pending, latched, and gait-edge state for selected environments."""
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._pending_offsets[env_ids] = 0.0
    self._latched_offsets[env_ids] = 0.0
    self._was_swinging[env_ids] = False
