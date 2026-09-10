"""State conversion between mjlab tensors and the legacy G23 MPC controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

G23_MPC_JOINT_ORDER = (
  "FL_HipX_joint",
  "FL_HipY_joint",
  "FL_Knee_joint",
  "FR_HipX_joint",
  "FR_HipY_joint",
  "FR_Knee_joint",
  "HL_HipX_joint",
  "HL_HipY_joint",
  "HL_Knee_joint",
  "HR_HipX_joint",
  "HR_HipY_joint",
  "HR_Knee_joint",
)


@dataclass(frozen=True)
class MpcControllerBatch:
  """CPU arrays consumed by the per-environment legacy controller."""

  dof_states: np.ndarray
  body_states: np.ndarray
  commands: np.ndarray
  foot_placement_offsets: np.ndarray
  foot_contacts: np.ndarray
  terrain_height_samples: np.ndarray | None


def pack_mpc_controller_batch(
  *,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  root_pos_w: torch.Tensor,
  root_quat_wxyz: torch.Tensor,
  root_lin_vel_w: torch.Tensor,
  root_ang_vel_w: torch.Tensor,
  env_origins: torch.Tensor,
  commands: torch.Tensor,
  foot_placement_offsets: torch.Tensor,
  foot_contacts: torch.Tensor,
  terrain_height_samples: torch.Tensor | None = None,
) -> MpcControllerBatch:
  """Pack one synchronized device-to-host transfer for the legacy controller.

  mjlab stores quaternions as ``wxyz`` and places environments at terrain-grid
  origins.  The legacy controller expects local positions and ``xyzw``.
  """
  num_envs = joint_pos.shape[0]
  expected = {
    "joint_pos": (num_envs, 12),
    "joint_vel": (num_envs, 12),
    "root_pos_w": (num_envs, 3),
    "root_quat_wxyz": (num_envs, 4),
    "root_lin_vel_w": (num_envs, 3),
    "root_ang_vel_w": (num_envs, 3),
    "env_origins": (num_envs, 3),
    "commands": (num_envs, 3),
    "foot_placement_offsets": (num_envs, 4, 2),
    "foot_contacts": (num_envs, 4),
  }
  tensors = {
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "root_pos_w": root_pos_w,
    "root_quat_wxyz": root_quat_wxyz,
    "root_lin_vel_w": root_lin_vel_w,
    "root_ang_vel_w": root_ang_vel_w,
    "env_origins": env_origins,
    "commands": commands,
    "foot_placement_offsets": foot_placement_offsets,
    "foot_contacts": foot_contacts,
  }
  if terrain_height_samples is not None:
    if (
      terrain_height_samples.ndim != 3
      or terrain_height_samples.shape[0] != num_envs
      or terrain_height_samples.shape[1] == 0
      or terrain_height_samples.shape[2] != 3
    ):
      raise ValueError(
        "terrain_height_samples must have shape "
        f"({num_envs}, num_samples, 3), got "
        f"{tuple(terrain_height_samples.shape)}"
      )
    tensors["terrain_height_samples"] = terrain_height_samples
    expected["terrain_height_samples"] = tuple(terrain_height_samples.shape)
  for name, tensor in tensors.items():
    if tensor.shape != expected[name]:
      raise ValueError(
        f"{name} must have shape {expected[name]}, got {tuple(tensor.shape)}"
      )
    if tensor.device != joint_pos.device:
      raise ValueError(f"{name} must be on device {joint_pos.device}")

  local_root_pos = root_pos_w - env_origins
  root_quat_xyzw = torch.cat((root_quat_wxyz[:, 1:4], root_quat_wxyz[:, 0:1]), dim=1)
  chunks = [
    joint_pos,
    joint_vel,
    local_root_pos,
    root_quat_xyzw,
    root_lin_vel_w,
    root_ang_vel_w,
    commands,
    foot_placement_offsets.flatten(start_dim=1),
    foot_contacts.to(dtype=joint_pos.dtype),
  ]
  if terrain_height_samples is not None:
    chunks.append(terrain_height_samples.flatten(start_dim=1))
  packed = torch.cat(chunks, dim=1)
  host = packed.detach().to(device="cpu", dtype=torch.float32).numpy()
  if not np.all(np.isfinite(host)):
    raise ValueError("MPC controller input contains non-finite values")

  dof_states = np.stack((host[:, 0:12], host[:, 12:24]), axis=-1)
  host_terrain_samples = None
  if terrain_height_samples is not None:
    host_terrain_samples = np.ascontiguousarray(
      host[:, 52:].reshape(num_envs, terrain_height_samples.shape[1], 3)
    )
  return MpcControllerBatch(
    dof_states=np.ascontiguousarray(dof_states),
    body_states=np.ascontiguousarray(host[:, 24:37]),
    commands=np.ascontiguousarray(host[:, 37:40]),
    foot_placement_offsets=np.ascontiguousarray(host[:, 40:48].reshape(num_envs, 4, 2)),
    foot_contacts=np.ascontiguousarray(host[:, 48:52] > 0),
    terrain_height_samples=host_terrain_samples,
  )
