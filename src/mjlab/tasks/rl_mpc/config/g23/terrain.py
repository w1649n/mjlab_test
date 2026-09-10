"""Terrain configuration for the SyncAI G23 RL-MPC task.

The terrain contract is intentionally narrower than the generic velocity-task
terrain set:

* a dedicated flat column is always available;
* uneven columns are continuous heightfields, never stair generators; and
* ``MAX_TERRAIN_HEIGHT_SPAN_M`` is a peak-to-peak limit, not an amplitude.

Keeping this contract in a small, separately tested module lets the pure-MPC
validation run use exactly the same terrain bank as the later RL task.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from mjlab.terrains import (
  BoxFlatTerrainCfg,
  HfRandomUniformTerrainCfg,
  HfWaveTerrainCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import TerrainOutput

MAX_TERRAIN_HEIGHT_SPAN_M = 0.08
"""Maximum peak-to-peak terrain height difference, in meters."""

_HEIGHTFIELD_HORIZONTAL_SCALE_M = 0.10
_HEIGHTFIELD_VERTICAL_SCALE_M = 0.005


class _FlatAtZeroCappedRandomUniformTerrainCfg(HfRandomUniformTerrainCfg):
  """Generate smooth random terrain while enforcing its physical height span."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    if difficulty <= 0.0:
      return BoxFlatTerrainCfg(size=self.size).function(difficulty, spec, rng)
    output = super().function(difficulty, spec, rng)
    # Cubic upsampling can overshoot the configured source-noise range. Keep
    # the smooth shape but rescale its physical heightfield extent to the hard
    # task limit instead of falling back to blocky cell-wise noise.
    for geometry in output.geometries:
      if geometry.hfield is not None:
        geometry.hfield.size[2] = min(
          geometry.hfield.size[2], MAX_TERRAIN_HEIGHT_SPAN_M
        )
    return output


class _FlatAtZeroWaveTerrainCfg(HfWaveTerrainCfg):
  """Use a real plane at curriculum level zero, not a degenerate heightfield."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    if difficulty <= 0.0:
      return BoxFlatTerrainCfg(size=self.size).function(difficulty, spec, rng)
    return super().function(difficulty, spec, rng)


RLMPC_G23_TERRAINS_CFG = TerrainGeneratorCfg(
  seed=42,
  size=(8.0, 8.0),
  border_width=10.0,
  num_rows=6,
  # In curriculum mode there is exactly one column per sub-terrain type.
  num_cols=3,
  curriculum=True,
  difficulty_range=(0.0, 1.0),
  sub_terrains={
    "flat": BoxFlatTerrainCfg(proportion=0.40),
    "random_rough": _FlatAtZeroCappedRandomUniformTerrainCfg(
      proportion=0.40,
      noise_range=(0.0, MAX_TERRAIN_HEIGHT_SPAN_M),
      noise_step=0.01,
      downsampled_scale=0.40,
      horizontal_scale=_HEIGHTFIELD_HORIZONTAL_SCALE_M,
      vertical_scale=_HEIGHTFIELD_VERTICAL_SCALE_M,
      border_width=0.50,
      scale_with_difficulty=True,
    ),
    "low_wave": _FlatAtZeroWaveTerrainCfg(
      proportion=0.20,
      # HfWaveTerrainCfg's generated peak-to-peak span is twice its
      # configured amplitude at full difficulty. Its integer height grid uses
      # half-amplitude units, so 1 cm is the smallest non-zero amplitude at a
      # 5 mm vertical scale. Level zero still returns a real plane above.
      amplitude_range=(
        2.0 * _HEIGHTFIELD_VERTICAL_SCALE_M,
        MAX_TERRAIN_HEIGHT_SPAN_M / 2.0,
      ),
      num_waves=3,
      horizontal_scale=_HEIGHTFIELD_HORIZONTAL_SCALE_M,
      vertical_scale=_HEIGHTFIELD_VERTICAL_SCALE_M,
      border_width=0.50,
    ),
  },
  add_lights=True,
)


def validate_rlmpc_g23_terrain_cfg(cfg: TerrainGeneratorCfg) -> None:
  """Validate the no-stairs, 8 cm peak-to-peak terrain contract.

  The dedicated capped random-terrain subclass may use coarse source samples
  for spatial smoothness because it clamps the compiled physical height span
  after spline interpolation. Other random configurations must avoid possible
  interpolation overshoot.
  """
  if not cfg.curriculum:
    raise ValueError("G23 RL-MPC terrain columns must map deterministically to types")

  difficulty_min, difficulty_max = cfg.difficulty_range
  if not 0.0 <= difficulty_min <= difficulty_max <= 1.0:
    raise ValueError("Terrain difficulty must stay within [0, 1]")

  proportions = [terrain.proportion for terrain in cfg.sub_terrains.values()]
  if abs(sum(proportions) - 1.0) > 1e-9:
    raise ValueError("G23 RL-MPC terrain proportions must sum to 1")

  if not any(
    isinstance(terrain, BoxFlatTerrainCfg) for terrain in cfg.sub_terrains.values()
  ):
    raise ValueError("G23 RL-MPC terrain set must contain an ordinary flat terrain")

  for name, terrain in cfg.sub_terrains.items():
    type_name = type(terrain).__name__
    if "stair" in name.lower() or "stair" in type_name.lower():
      raise ValueError(f"Stair terrain is not allowed: {name} ({type_name})")

    if isinstance(terrain, BoxFlatTerrainCfg):
      continue

    if isinstance(terrain, HfRandomUniformTerrainCfg):
      configured_span = terrain.noise_range[1] - terrain.noise_range[0]
      if configured_span > MAX_TERRAIN_HEIGHT_SPAN_M + 1e-12:
        raise ValueError(f"Random terrain {name!r} exceeds the 8 cm height-span limit")
      if terrain.downsampled_scale != terrain.horizontal_scale and not isinstance(
        terrain, _FlatAtZeroCappedRandomUniformTerrainCfg
      ):
        raise ValueError(
          f"Random terrain {name!r} must disable interpolation overshoot"
        )
      continue

    if isinstance(terrain, HfWaveTerrainCfg):
      if terrain.amplitude_range[0] < 2.0 * terrain.vertical_scale:
        raise ValueError(
          f"Wave terrain {name!r} has positive levels that quantize to flat"
        )
      configured_span = 2.0 * max(abs(value) for value in terrain.amplitude_range)
      if configured_span > MAX_TERRAIN_HEIGHT_SPAN_M + 1e-12:
        raise ValueError(f"Wave terrain {name!r} exceeds the 8 cm height-span limit")
      continue

    raise ValueError(f"Unsupported G23 RL-MPC terrain type: {name} ({type_name})")


def make_rlmpc_g23_terrain_cfg() -> TerrainGeneratorCfg:
  """Return an independently mutable, validated terrain configuration."""
  cfg = copy.deepcopy(RLMPC_G23_TERRAINS_CFG)
  validate_rlmpc_g23_terrain_cfg(cfg)
  return cfg


# Fail at import time if a future edit weakens the task's terrain contract.
validate_rlmpc_g23_terrain_cfg(RLMPC_G23_TERRAINS_CFG)
