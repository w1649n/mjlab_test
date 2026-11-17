"""Convert assets from the mujoco-rl-nav repository into mjlab task data.

Reads the hospital obstacle rectangles, the STL they were derived from (for the
vertical extent of each obstacle) and the scenario functions, and writes:

- ``data/hospital_obstacles.csv``: one axis-aligned box per row.
- ``data/scenarios.npz``: a bank of pre-sampled start/goal/human layouts.

The scenario functions draw from ``np.random``, so each scenario is sampled many
times up front and the environment later picks rows from the bank on the GPU.

Only NumPy is required::

  python convert_assets.py /path/to/mujoco-rl-nav
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_MAX_HUMANS = 20
# Obstacles listed for the pedestrian model that have no geometry in the STL.
_DEFAULT_Z_RANGE = (0.0, 0.0)


def _parse_rectangles(path: Path) -> dict[str, np.ndarray]:
  """Parse ``Name:`` blocks of ``x, y`` corner lines into [4, 2] arrays."""
  rects: dict[str, list[list[float]]] = {}
  name = ""
  for raw_line in path.read_text().splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if line.endswith(":"):
      name = line[:-1]
      rects[name] = []
    else:
      rects[name].append([float(v) for v in line.split(",")])
  return {k: np.asarray(v, dtype=np.float64) for k, v in rects.items()}


def _read_binary_stl_vertices(path: Path) -> np.ndarray:
  raw = path.read_bytes()
  num_tris = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
  if 84 + 50 * num_tris != len(raw):
    raise ValueError(f"{path} is not a binary STL")
  dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
  tris = np.frombuffer(raw[84:], dtype=dtype)["v"]
  return tris.reshape(-1, 3).astype(np.float64)


def convert_obstacles(repo: Path, out_path: Path) -> np.ndarray:
  """Write the obstacle table and return the map bounds (xmin, ymin, xmax, ymax)."""
  rects = _parse_rectangles(repo / "src/utils/grid_decomp/mesh_edges.txt")
  verts = _read_binary_stl_vertices(repo / "assets/Floor_hospital_new.stl")
  rows = ["name,xmin,ymin,xmax,ymax,zmin,zmax"]
  for name in sorted(rects):
    corners = rects[name]
    at_corner = np.zeros(len(verts), dtype=bool)
    for corner in corners:
      at_corner |= np.abs(verts[:, :2] - corner).max(axis=1) < 2e-3
    if at_corner.any():
      zmin, zmax = verts[at_corner, 2].min(), verts[at_corner, 2].max()
    else:
      zmin, zmax = _DEFAULT_Z_RANGE
    (xmin, ymin), (xmax, ymax) = corners.min(axis=0), corners.max(axis=0)
    rows.append(
      f"{name},{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f},{zmin:.4f},{zmax:.4f}"
    )
  out_path.write_text("\n".join(rows) + "\n")
  print(f"Wrote {len(rows) - 1} obstacles.")
  corners = np.concatenate(list(rects.values()))
  return np.concatenate([corners.min(axis=0), corners.max(axis=0)])


def _load_scenario_fns(repo: Path) -> dict[str, types.FunctionType]:
  # The scenario modules import simulator packages they never use.
  for stub in ("gymnasium", "mujoco", "mujoco.viewer"):
    sys.modules.setdefault(stub, types.ModuleType(stub))
  # Some scenarios import ``env_config`` from the repo's core package.
  sys.path.insert(0, str(repo / "src/core"))
  fns: dict[str, types.FunctionType] = {}
  for path in sorted((repo / "data/scenarios").glob("scenario*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fns[path.stem] = getattr(module, path.stem)
  return fns


def convert_scenarios(
  repo: Path, out_path: Path, bounds: np.ndarray, samples: int, seed: int
) -> list[str]:
  fns = _load_scenario_fns(repo)
  names = list(fns)
  shape = (len(names), samples)
  robot_start = np.zeros((*shape, 3), dtype=np.float32)
  goal = np.zeros((*shape, 2), dtype=np.float32)
  human_pos = np.zeros((*shape, _MAX_HUMANS, 2), dtype=np.float32)
  human_yaw = np.zeros((*shape, _MAX_HUMANS), dtype=np.float32)
  human_goal = np.zeros((*shape, _MAX_HUMANS, 2), dtype=np.float32)
  human_active = np.zeros((*shape, _MAX_HUMANS), dtype=bool)

  np.random.seed(seed)
  for i, name in enumerate(names):
    for j in range(samples):
      d = fns[name]()
      robot_start[i, j] = (
        d["mob_robot_startposx"],
        d["mob_robot_startposy"],
        d["mob_robot_start_orientation"],
      )
      goal[i, j] = (d["target_robot_x"], d["target_robot_y"])
      for h in range(_MAX_HUMANS):
        if f"human{h + 1}x" not in d:
          continue
        start = np.array([d[f"human{h + 1}x"], d[f"human{h + 1}y"]])
        # Scenarios park the pedestrians they do not use outside the building.
        if np.any(start < bounds[:2]) or np.any(start > bounds[2:]):
          continue
        human_active[i, j, h] = True
        human_pos[i, j, h] = start
        human_goal[i, j, h] = (d[f"targethuman{h + 1}x"], d[f"targethuman{h + 1}y"])
        # Scenario files give pedestrian headings in degrees.
        human_yaw[i, j, h] = np.deg2rad(d[f"start_orientation_human{h + 1}"])

  robot_start[..., 2] = (robot_start[..., 2] + np.pi) % (2 * np.pi) - np.pi
  human_yaw = (human_yaw + np.pi) % (2 * np.pi) - np.pi
  np.savez_compressed(
    out_path,
    names=np.asarray(names),
    robot_start=robot_start,
    goal=goal,
    human_pos=human_pos,
    human_yaw=human_yaw,
    human_goal=human_goal,
    human_active=human_active,
  )
  return names


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("repo", type=Path, help="Path to the mujoco-rl-nav checkout.")
  parser.add_argument("--samples", type=int, default=2048)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  _DATA_DIR.mkdir(parents=True, exist_ok=True)
  bounds = convert_obstacles(args.repo, _DATA_DIR / "hospital_obstacles.csv")
  names = convert_scenarios(
    args.repo, _DATA_DIR / "scenarios.npz", bounds, args.samples, args.seed
  )
  print(f"Wrote {args.samples} samples for each of {len(names)} scenarios: {names}")


if __name__ == "__main__":
  main()
