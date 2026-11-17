"""Batched Headed Social Force Model (HSFM) for pedestrians.

A PyTorch port of the JAX model used by the reference navigation environment,
vectorized over environments and pedestrians. Obstacles are axis-aligned boxes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1e-3


@dataclass
class HsfmParams:
  """HSFM parameters, shared by every pedestrian."""

  radius: float = 0.3
  mass: float = 75.0
  max_speed: float = 1.0
  relaxation_time: float = 0.5
  social_strength: float = 2000.0
  """Repulsion magnitude between pedestrians (Ai)."""
  obstacle_strength: float = 2000.0
  """Repulsion magnitude from obstacles (Aw)."""
  social_range: float = 0.08
  """Repulsion length scale between pedestrians (Bi)."""
  obstacle_range: float = 0.08
  """Repulsion length scale from obstacles (Bw)."""
  social_sliding_strength: float = 120.0
  """Sliding force magnitude between pedestrians (Ci)."""
  obstacle_sliding_strength: float = 120.0
  """Sliding force magnitude along obstacles (Cw)."""
  social_sliding_range: float = 0.6
  """Sliding force length scale between pedestrians (Di)."""
  obstacle_sliding_range: float = 0.6
  """Sliding force length scale along obstacles (Dw)."""
  body_stiffness: float = 120000.0
  """Compression stiffness once bodies overlap (k1)."""
  friction_stiffness: float = 240000.0
  """Sliding friction once bodies overlap (k2)."""
  lateral_gain: float = 1.0
  """Gain on the sideways component of the repulsive forces (ko)."""
  lateral_damping: float = 500.0
  """Damping of the sideways body velocity (kd)."""
  alpha: float = 3.0
  k_lambda: float = 0.1
  safety_space: float = 0.0
  obstacle_neighborhood: float = 4.0
  """Obstacles closer than this share the obstacle force, like the reference
  model's grid-cell lookup."""


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
  return (theta + torch.pi) % (2 * torch.pi) - torch.pi


def _world_velocity(yaw: torch.Tensor, body_vel: torch.Tensor) -> torch.Tensor:
  cos, sin = torch.cos(yaw), torch.sin(yaw)
  return torch.stack(
    [
      cos * body_vel[..., 0] - sin * body_vel[..., 1],
      sin * body_vel[..., 0] + cos * body_vel[..., 1],
    ],
    dim=-1,
  )


def _perpendicular(v: torch.Tensor) -> torch.Tensor:
  return torch.stack([-v[..., 1], v[..., 0]], dim=-1)


def hsfm_step(
  state: torch.Tensor,
  goal: torch.Tensor,
  active: torch.Tensor,
  boxes: torch.Tensor,
  params: HsfmParams,
  dt: float,
  robot_pos: torch.Tensor | None = None,
  robot_vel: torch.Tensor | None = None,
  robot_radius: float = 0.0,
) -> torch.Tensor:
  """Advance every pedestrian by ``dt``.

  Args:
    state: [B, H, 6] rows of (px, py, body vx, body vy, yaw, yaw rate).
    goal: [B, H, 2] current goal of each pedestrian.
    active: [B, H] pedestrians that take part in the simulation.
    boxes: [N, 4] obstacle footprints (xmin, ymin, xmax, ymax).
    params: Model parameters.
    dt: Integration step in seconds.
    robot_pos: Optional [B, 2] robot position. When given, pedestrians avoid
      the robot as if it were another pedestrian.
    robot_vel: [B, 2] world-frame robot velocity, required with ``robot_pos``.
    robot_radius: Radius of the robot.

  Returns:
    The updated [B, H, 6] state. Inactive pedestrians are left untouched.
  """
  p = params
  pos, body_vel = state[..., 0:2], state[..., 2:4]
  yaw, yaw_rate = state[..., 4], state[..., 5]
  vel = _world_velocity(yaw, body_vel)

  # Driving force towards the goal.
  to_goal = goal - pos
  goal_dist = torch.linalg.norm(to_goal, dim=-1, keepdim=True)
  desired_vel = to_goal / goal_dist.clamp(min=_EPS) * p.max_speed
  desired_force = p.mass * (desired_vel - vel) / p.relaxation_time
  desired_force = torch.where(goal_dist > p.radius, desired_force, 0.0)

  # Repulsion from the other agents: [B, H, K] with K = H (+ 1 for the robot).
  other_pos, other_vel = pos, vel
  other_active = active
  other_radius = torch.full_like(yaw[:1, :], p.radius)
  if robot_pos is not None:
    assert robot_vel is not None
    other_pos = torch.cat([pos, robot_pos[:, None]], dim=1)
    other_vel = torch.cat([vel, robot_vel[:, None]], dim=1)
    other_active = torch.cat([active, torch.ones_like(active[:, :1])], dim=1)
    other_radius = torch.cat(
      [other_radius, torch.full_like(other_radius[:, :1], robot_radius)], dim=1
    )
  num_humans = pos.shape[1]
  diff = pos[:, :, None] - other_pos[:, None, :]
  dist = torch.linalg.norm(diff, dim=-1)
  normal = diff / (dist[..., None] + _EPS)
  tangent = _perpendicular(normal)
  overlap = p.radius + other_radius[:, None, :] + 2 * p.safety_space - dist
  compression = overlap.clamp(min=0.0)
  sliding_vel = ((other_vel[:, None, :] - vel[:, :, None]) * tangent).sum(dim=-1)
  social = (
    p.social_strength * torch.exp(overlap / p.social_range)
    + p.body_stiffness * compression
  )[..., None] * normal + (
    p.social_sliding_strength * torch.exp(overlap / p.social_sliding_range)
    + p.friction_stiffness * compression * sliding_vel
  )[..., None] * tangent
  pair_mask = other_active[:, None, :].expand(-1, num_humans, -1).clone()
  pair_mask[:, :, :num_humans] &= ~torch.eye(
    num_humans, dtype=torch.bool, device=state.device
  )
  social_force = (social * pair_mask[..., None]).sum(dim=2)

  # Repulsion from the closest point of each nearby obstacle: [B, H, N].
  closest = torch.maximum(
    torch.minimum(pos[:, :, None], boxes[None, None, :, 2:4]), boxes[None, None, :, 0:2]
  )
  diff = pos[:, :, None] - closest
  dist = torch.linalg.norm(diff, dim=-1)
  normal = diff / (dist[..., None] + _EPS)
  tangent = _perpendicular(normal)
  overlap = p.radius + p.safety_space - dist
  compression = overlap.clamp(min=0.0)
  sliding_vel = -(vel[:, :, None] * tangent).sum(dim=-1)
  obstacle = (
    p.obstacle_strength * torch.exp(overlap / p.obstacle_range)
    + p.body_stiffness * compression
  )[..., None] * normal - (
    (
      p.obstacle_sliding_strength * torch.exp(overlap / p.obstacle_sliding_range)
      + p.friction_stiffness * compression
    )
    * sliding_vel
  )[..., None] * tangent
  nearby = dist < p.obstacle_neighborhood
  obstacle_force = (obstacle * nearby[..., None]).sum(dim=2)
  obstacle_force = obstacle_force / nearby.sum(dim=-1, keepdim=True).clamp(min=1)

  # Torque that turns the pedestrian towards the total force.
  repulsive_force = social_force + obstacle_force
  total_force = desired_force + repulsive_force
  force_norm = torch.linalg.norm(total_force, dim=-1)
  force_yaw = torch.atan2(total_force[..., 1], total_force[..., 0])
  inertia = p.mass * p.radius**2 / 2
  k_theta = inertia * p.k_lambda * force_norm
  k_omega = inertia * (1 + p.alpha) * torch.sqrt(p.k_lambda * force_norm / p.alpha)
  torque = -k_theta * wrap_angle(yaw - force_yaw) - k_omega * yaw_rate
  torque = torque.clamp(-100.0, 100.0)

  # Forces in the body frame: full force forwards, damped repulsion sideways.
  forward = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
  sideways = _perpendicular(forward)
  body_force = torch.stack(
    [
      (total_force * forward).sum(dim=-1),
      p.lateral_gain * (repulsive_force * sideways).sum(dim=-1)
      - p.lateral_damping * body_vel[..., 1],
    ],
    dim=-1,
  )

  new_pos = pos + dt * vel
  new_yaw = wrap_angle(yaw + dt * yaw_rate)
  new_body_vel = body_vel + dt * body_force / p.mass
  speed = torch.linalg.norm(new_body_vel, dim=-1, keepdim=True)
  new_body_vel = new_body_vel * (p.max_speed / speed.clamp(min=p.max_speed))
  new_yaw_rate = (yaw_rate + dt * torque / inertia).clamp(-10.0, 10.0)

  new_state = torch.cat(
    [new_pos, new_body_vel, new_yaw[..., None], new_yaw_rate[..., None]], dim=-1
  )
  new_state = torch.nan_to_num(new_state, nan=0.0, posinf=1e3, neginf=-1e3)
  return torch.where(active[..., None], new_state, state)
