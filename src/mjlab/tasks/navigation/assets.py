"""Scene assets for the hospital navigation task.

The hospital floor plan is a set of axis-aligned boxes (walls and tables), the
robot is a planar unicycle base and pedestrians are cylinders. Everything moves
kinematically, so no geom takes part in contact.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import EntityCfg

DATA_DIR: Path = Path(__file__).parent / "data"

ROBOT_RADIUS = 0.2
HUMAN_RADIUS = 0.3
MAX_HUMANS = 20
LIDAR_HEIGHT = 1.0

# Pedestrians that a scenario does not use wait here, far outside the map.
PARKING_ORIGIN = (-150.0, -150.0)
PARKING_SPACING = 2.0

_WALL_RGBA = (0.5, 0.5, 0.7, 1.0)
_TABLE_RGBA = (0.65, 0.5, 0.35, 1.0)
_HUMAN_HALF_HEIGHT = 0.9


@dataclass(frozen=True)
class HospitalMap:
  """Axis-aligned obstacle boxes of the hospital floor."""

  names: tuple[str, ...]
  boxes: np.ndarray
  """[N, 6] rows of (xmin, ymin, xmax, ymax, zmin, zmax)."""

  @property
  def has_geometry(self) -> np.ndarray:
    """Boxes with a vertical extent. The rest only exist for pedestrians."""
    return self.boxes[:, 5] > self.boxes[:, 4]

  def boxes_at_height(self, height: float) -> np.ndarray:
    """[M, 4] footprints (xmin, ymin, xmax, ymax) of boxes spanning ``height``."""
    mask = (self.boxes[:, 4] <= height) & (self.boxes[:, 5] >= height)
    return self.boxes[mask, :4]


def load_hospital_map() -> HospitalMap:
  with open(DATA_DIR / "hospital_obstacles.csv", newline="") as f:
    rows = list(csv.DictReader(f))
  keys = ("xmin", "ymin", "xmax", "ymax", "zmin", "zmax")
  boxes = np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)
  return HospitalMap(names=tuple(r["name"] for r in rows), boxes=boxes)


def _disable_contact(geom: mujoco.MjsGeom) -> None:
  geom.contype = 0
  geom.conaffinity = 0


def get_hospital_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "hospital"
  body = spec.worldbody.add_body(name="hospital")
  hospital_map = load_hospital_map()
  for name, box, has_geometry in zip(
    hospital_map.names, hospital_map.boxes, hospital_map.has_geometry, strict=True
  ):
    if not has_geometry:
      continue
    xmin, ymin, xmax, ymax, _, zmax = box
    # Boxes start at the floor so nothing floats or sinks below the ground.
    geom = body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=((xmin + xmax) / 2, (ymin + ymax) / 2, zmax / 2),
      size=((xmax - xmin) / 2, (ymax - ymin) / 2, zmax / 2),
      rgba=_TABLE_RGBA if name.startswith("Table") else _WALL_RGBA,
    )
    _disable_contact(geom)
  return spec


def get_robot_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "robot"
  body = spec.worldbody.add_body(name="base")
  body.add_joint(name="x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(1, 0, 0))
  body.add_joint(name="y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(0, 1, 0))
  body.add_joint(name="yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1))
  chassis = body.add_geom(
    name="chassis",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(ROBOT_RADIUS, 0.13, 0),
    pos=(0, 0, 0.13),
    rgba=(0.15, 0.15, 0.15, 1.0),
  )
  heading = body.add_geom(
    name="heading",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.12, 0.03, 0.01),
    pos=(0.1, 0, 0.27),
    rgba=(0.1, 0.9, 0.1, 1.0),
  )
  mast = body.add_geom(
    name="mast",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(0.02, (LIDAR_HEIGHT - 0.26) / 2, 0),
    pos=(0, 0, (LIDAR_HEIGHT + 0.26) / 2),
    rgba=(0.4, 0.4, 0.4, 1.0),
  )
  for geom in (chassis, heading, mast):
    _disable_contact(geom)
  body.add_site(name="lidar", pos=(0, 0, LIDAR_HEIGHT), size=(0.03, 0.03, 0.03))
  return spec


def get_humans_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "humans"
  # A static root keeps every pedestrian inside one fixed-base entity.
  root = spec.worldbody.add_body(name="humans")
  for i in range(MAX_HUMANS):
    hue = i / MAX_HUMANS
    rgba = (0.3 + 0.7 * hue, 0.8 - 0.6 * hue, 0.3 + 0.5 * ((i * 7) % 10) / 10, 1.0)
    body = root.add_body(name=f"human_{i}")
    body.add_joint(
      name=f"human_{i}_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(1, 0, 0)
    )
    body.add_joint(
      name=f"human_{i}_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(0, 1, 0)
    )
    body.add_joint(
      name=f"human_{i}_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1)
    )
    torso = body.add_geom(
      name=f"human_{i}_torso",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(HUMAN_RADIUS, _HUMAN_HALF_HEIGHT, 0),
      pos=(0, 0, _HUMAN_HALF_HEIGHT),
      rgba=rgba,
    )
    nose = body.add_geom(
      name=f"human_{i}_nose",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.08, 0.04, 0.04),
      pos=(HUMAN_RADIUS, 0, 1.6),
      rgba=(0.1, 0.1, 0.1, 1.0),
    )
    for geom in (torso, nose):
      _disable_contact(geom)
  return spec


def parking_positions() -> np.ndarray:
  """[MAX_HUMANS, 2] positions for pedestrians that are not in use."""
  xs = PARKING_ORIGIN[0] - PARKING_SPACING * np.arange(MAX_HUMANS)
  ys = np.full(MAX_HUMANS, PARKING_ORIGIN[1])
  return np.stack([xs, ys], axis=-1)


def get_hospital_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=get_hospital_spec)


def get_robot_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=get_robot_spec)


def get_humans_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=get_humans_spec)
