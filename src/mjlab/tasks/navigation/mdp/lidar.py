"""Planar lidar ray pattern for :class:`mjlab.sensor.RayCastSensor`."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import torch


@dataclass
class PlanarLidarPatternCfg:
  """Rays fanned out over a full turn in the frame's XY plane.

  Ray ``i`` points at ``2 * pi * i / num_rays`` from the frame's X axis, so ray
  0 looks straight ahead and the angle grows counter-clockwise.
  """

  num_rays: int = 108

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (local_offsets [N, 3], local_directions [N, 3])."""
    del mj_model
    angles = torch.arange(self.num_rays, device=device) * (
      2.0 * math.pi / self.num_rays
    )
    directions = torch.stack(
      [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=-1
    )
    return torch.zeros_like(directions), directions
