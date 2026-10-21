"""Left-right mirror augmentation for the SyncAI G23 velocity tasks.

The mirror reflects every quantity across the base's sagittal (x-z) plane: body
y components flip, roll and yaw rates flip, left and right legs swap, and the
abduction (HipX) joints change sign. HipY and knee axes are the same on both
sides of the G23 MJCF, so those joints only swap legs.

Pass ``"mjlab.tasks.velocity.config.g23.symmetry:mirror_g23_velocity"`` as the
``data_augmentation_func`` of the PPO ``symmetry_cfg``.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from tensordict import TensorDict

from mjlab.sensor import ContactSensor, ObjRef, RayCastSensor
from mjlab.tasks.velocity import mdp

_LEG_MIRROR = {"FL": "FR", "FR": "FL", "HL": "HR", "HR": "HL"}

# Per-component signs of base-frame 3-vectors under the mirror.
_VECTOR_SIGNS: dict[str, tuple[float, float, float]] = {
  "base_lin_vel": (1.0, -1.0, 1.0),
  "base_ang_vel": (-1.0, 1.0, -1.0),
  "projected_gravity": (1.0, -1.0, 1.0),
  # (vx, vy, wz): lateral velocity and yaw rate flip.
  "command": (1.0, -1.0, -1.0),
}
_JOINT_TERMS = ("joint_pos", "joint_vel")
_FOOT_SENSOR_TERMS = ("foot_height", "foot_air_time", "foot_contact")


@dataclass(frozen=True)
class Mirror:
  """``mirrored[..., i] = sign[i] * x[..., perm[i]]`` for one observation frame."""

  perm: tuple[int, ...]
  sign: tuple[float, ...]

  def tile(self, frames: int) -> Mirror:
    """Repeat the mirror over ``frames`` consecutive (flattened history) frames."""
    size = len(self.perm)
    perm = tuple(f * size + p for f in range(frames) for p in self.perm)
    return Mirror(perm, self.sign * frames)


def concat_mirrors(mirrors: Sequence[Mirror]) -> Mirror:
  """Mirror for terms concatenated in order."""
  perm: list[int] = []
  sign: list[float] = []
  offset = 0
  for mirror in mirrors:
    perm.extend(offset + p for p in mirror.perm)
    sign.extend(mirror.sign)
    offset += len(mirror.perm)
  return Mirror(tuple(perm), tuple(sign))


def _leg(name: str) -> str:
  leg = name.split("_", 1)[0]
  if leg not in _LEG_MIRROR:
    raise ValueError(f"Cannot tell the leg of '{name}'.")
  return leg


def joint_mirror(names: Sequence[str]) -> Mirror:
  """Swap left/right legs and negate abduction for ``<leg>_<joint>`` names."""
  index = {name: i for i, name in enumerate(names)}
  perm, sign = [], []
  for name in names:
    leg, joint = name.split("_", 1)
    perm.append(index[f"{_LEG_MIRROR[_leg(name)]}_{joint}"])
    sign.append(-1.0 if joint.startswith("HipX") else 1.0)
  return Mirror(tuple(perm), tuple(sign))


def foot_mirror(names: Sequence[str], block_signs: Sequence[float] = (1.0,)) -> Mirror:
  """Swap per-foot blocks; ``block_signs`` applies within each foot's block."""
  legs = [_leg(name) for name in names]
  block = len(block_signs)
  perm, sign = [], []
  for leg in legs:
    other = legs.index(_LEG_MIRROR[leg])
    perm.extend(other * block + k for k in range(block))
    sign.extend(block_signs)
  return Mirror(tuple(perm), tuple(sign))


def grid_mirror(offsets: torch.Tensor, frames: int = 1) -> Mirror:
  """Map each ray at local (x, y) to the ray at (x, -y)."""
  xy = offsets[:, :2].double()
  target = xy * torch.tensor([1.0, -1.0], dtype=xy.dtype)
  dist = torch.cdist(target, xy)
  nearest = dist.argmin(dim=1)
  if dist.gather(1, nearest[:, None]).max() > 1e-4:
    raise ValueError("Ray pattern is not symmetric about the x axis.")
  return Mirror(tuple(nearest.tolist()), (1.0,) * len(nearest)).tile(frames)


def _frame_names(sensor: RayCastSensor) -> list[str]:
  frame = sensor.cfg.frame
  frames = (frame,) if isinstance(frame, ObjRef) else frame
  return [ref.name for ref in frames]


def _require_single_slot(sensor: ContactSensor) -> None:
  # Extra contact slots would be indistinguishable from history frames below.
  if sensor.cfg.reduce != "netforce" and sensor.cfg.num_slots != 1:
    raise ValueError(f"Contact sensor '{sensor.cfg.name}' must report one slot.")


def _term_mirror(env: Any, group: str, name: str, dim: int) -> Mirror:
  """Mirror for one (possibly history-flattened) observation term."""
  term_cfg = env.cfg.observations[group].terms[name]
  params = term_cfg.params
  if name in _VECTOR_SIGNS:
    base = Mirror((0, 1, 2), _VECTOR_SIGNS[name])
  elif name in _JOINT_TERMS:
    base = joint_mirror(env.scene["robot"].joint_names)
  elif name == "actions":
    base = _action_mirror(env)
  elif name == "height_scan":
    sensor = env.scene[params["sensor_name"]]
    offsets, _ = sensor.cfg.pattern.generate_rays(None, "cpu")
    base = grid_mirror(offsets, sensor.num_frames)
  elif name in _FOOT_SENSOR_TERMS:
    sensor = env.scene[params["sensor_name"]]
    if isinstance(sensor, ContactSensor):
      _require_single_slot(sensor)
      base = foot_mirror(sensor.primary_names)
    else:
      base = foot_mirror(_frame_names(sensor))
  elif name == "foot_contact_forces":
    if term_cfg.func is not mdp.foot_contact_forces_b:
      raise ValueError(
        "World-frame foot_contact_forces cannot be mirrored without the heading; "
        "use mdp.foot_contact_forces_b."
      )
    sensor = env.scene[params["sensor_name"]]
    _require_single_slot(sensor)
    base = foot_mirror(sensor.primary_names, (1.0, -1.0, 1.0))
  else:
    raise ValueError(f"No mirror defined for observation term '{group}/{name}'.")
  frames, remainder = divmod(dim, len(base.perm))
  if remainder:
    raise ValueError(
      f"Term '{group}/{name}' has {dim} values, not a multiple of {len(base.perm)}."
    )
  return base.tile(frames)


def _action_mirror(env: Any) -> Mirror:
  manager = env.action_manager
  names = [n for t in manager.active_terms for n in manager.get_term(t).target_names]
  return joint_mirror(names)


@dataclass(frozen=True)
class _MirrorSpec:
  obs: dict[str, tuple[torch.Tensor, torch.Tensor]]
  actions: tuple[torch.Tensor, torch.Tensor]


_SPECS: weakref.WeakKeyDictionary[Any, _MirrorSpec] = weakref.WeakKeyDictionary()


def _as_tensors(mirror: Mirror, device: str) -> tuple[torch.Tensor, torch.Tensor]:
  perm = torch.tensor(mirror.perm, dtype=torch.long, device=device)
  sign = torch.tensor(mirror.sign, dtype=torch.float, device=device)
  return perm, sign


def _build_spec(env: Any) -> _MirrorSpec:
  manager = env.observation_manager
  obs = {}
  for group, term_names in manager.active_terms.items():
    dims = manager.group_obs_term_dim[group]
    mirrors = [
      _term_mirror(env, group, name, int(torch.Size(dim).numel()))
      for name, dim in zip(term_names, dims, strict=True)
    ]
    obs[group] = _as_tensors(concat_mirrors(mirrors), env.device)
  return _MirrorSpec(obs, _as_tensors(_action_mirror(env), env.device))


def mirror_g23_velocity(
  env: Any,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append left-right mirrored copies of ``obs`` and ``actions`` along dim 0."""
  base_env = getattr(env, "unwrapped", env)
  spec = _SPECS.get(base_env)
  if spec is None:
    spec = _SPECS[base_env] = _build_spec(base_env)

  obs_aug = None
  if obs is not None:
    mirrored = {}
    for group, value in obs.items():
      perm, sign = spec.obs[group]
      mirrored[group] = value[..., perm] * sign
    mirrored_obs = TensorDict(mirrored, batch_size=obs.batch_size)
    obs_aug = cast(TensorDict, TensorDict.cat([obs, mirrored_obs], dim=0))
  actions_aug = None
  if actions is not None:
    perm, sign = spec.actions
    actions_aug = torch.cat([actions, actions[..., perm] * sign], dim=0)
  return obs_aug, actions_aug
