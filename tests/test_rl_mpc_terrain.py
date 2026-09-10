"""Contract tests for the SyncAI G23 RL-MPC terrain bank."""

import copy

import mujoco
import numpy as np
import pytest

from mjlab.tasks.rl_mpc.config.g23.terrain import (
  MAX_TERRAIN_HEIGHT_SPAN_M,
  RLMPC_G23_TERRAINS_CFG,
  make_rlmpc_g23_terrain_cfg,
  validate_rlmpc_g23_terrain_cfg,
)
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  HfRandomUniformTerrainCfg,
  TerrainGenerator,
)


def _generate_height_span(terrain, difficulty: float, seed: int) -> float:
  terrain = copy.deepcopy(terrain)
  terrain.size = (4.0, 4.0)
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = terrain.function(difficulty, spec, np.random.default_rng(seed))

  height_spans = []
  for geometry in output.geometries:
    if geometry.hfield is None:
      continue
    normalized_heights = np.asarray(geometry.hfield.userdata, dtype=np.float64)
    height_spans.append(float(np.ptp(normalized_heights) * geometry.hfield.size[2]))
  return max(height_spans, default=0.0)


def _generate_heightfield(terrain, difficulty: float, seed: int) -> np.ndarray:
  terrain = copy.deepcopy(terrain)
  terrain.size = (4.0, 4.0)
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = terrain.function(difficulty, spec, np.random.default_rng(seed))
  geometry = next(item for item in output.geometries if item.hfield is not None)
  hfield = geometry.hfield
  assert hfield is not None
  return (
    np.asarray(hfield.userdata, dtype=np.float64).reshape(hfield.nrow, hfield.ncol)
    * hfield.size[2]
  )


def test_rlmpc_g23_terrain_mix_contains_flat_and_no_stairs():
  cfg = make_rlmpc_g23_terrain_cfg()

  assert set(cfg.sub_terrains) == {"flat", "random_rough", "low_wave"}
  assert any(isinstance(t, BoxFlatTerrainCfg) for t in cfg.sub_terrains.values())
  assert abs(sum(t.proportion for t in cfg.sub_terrains.values()) - 1.0) < 1e-9

  for name, terrain in cfg.sub_terrains.items():
    assert "stair" not in name.lower()
    assert "stair" not in type(terrain).__name__.lower()


@pytest.mark.parametrize("difficulty", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_rlmpc_g23_generated_height_span_never_exceeds_8_cm(difficulty: float):
  cfg = make_rlmpc_g23_terrain_cfg()

  for name, terrain in cfg.sub_terrains.items():
    for seed in range(5):
      height_span = _generate_height_span(terrain, difficulty, seed)
      assert height_span <= MAX_TERRAIN_HEIGHT_SPAN_M + 1e-9, (
        f"{name} generated {height_span:.6f} m at difficulty={difficulty}, seed={seed}"
      )


def test_rlmpc_g23_uneven_terrains_reach_but_do_not_exceed_limit():
  cfg = make_rlmpc_g23_terrain_cfg()

  for name in ("random_rough", "low_wave"):
    height_span = _generate_height_span(cfg.sub_terrains[name], 1.0, seed=7)
    assert height_span == pytest.approx(MAX_TERRAIN_HEIGHT_SPAN_M)


def test_rlmpc_g23_every_positive_level_contains_real_uneven_terrain():
  cfg = make_rlmpc_g23_terrain_cfg()
  difficulties = [row / (cfg.num_rows - 1) for row in range(1, cfg.num_rows)]

  for name in ("random_rough", "low_wave"):
    spans = [
      _generate_height_span(cfg.sub_terrains[name], difficulty, seed=7)
      for difficulty in difficulties
    ]
    assert all(span > 0.0 for span in spans), f"{name} has a flat positive level"
    assert spans == sorted(spans), f"{name} height span is not monotonic: {spans}"


def test_rlmpc_g23_random_terrain_is_spatially_smooth():
  terrain = make_rlmpc_g23_terrain_cfg().sub_terrains["random_rough"]
  assert isinstance(terrain, HfRandomUniformTerrainCfg)
  heights = _generate_heightfield(terrain, 1.0, seed=7)
  border_pixels = int(terrain.border_width / terrain.horizontal_scale)
  interior = heights[
    border_pixels:-border_pixels,
    border_pixels:-border_pixels,
  ]
  max_neighbor_delta = max(
    float(np.max(np.abs(np.diff(interior, axis=0)))),
    float(np.max(np.abs(np.diff(interior, axis=1)))),
  )
  assert max_neighbor_delta <= 0.04


def test_rlmpc_g23_complete_terrain_bank_compiles():
  cfg = make_rlmpc_g23_terrain_cfg()
  generator = TerrainGenerator(cfg, device="cpu")
  spec = mujoco.MjSpec()
  generator.compile(spec)
  model = spec.compile()

  uneven_types = sum(
    not isinstance(terrain, BoxFlatTerrainCfg) for terrain in cfg.sub_terrains.values()
  )
  # Row zero of every curriculum column is an actual box plane; only the
  # remaining rows of uneven terrain types compile to heightfields.
  assert model.nhfield == (cfg.num_rows - 1) * uneven_types
  assert np.all(model.hfield_size[:, 2] <= MAX_TERRAIN_HEIGHT_SPAN_M + 1e-9)


def test_rlmpc_g23_level_zero_uses_plane_geometry_for_every_type():
  cfg = make_rlmpc_g23_terrain_cfg()

  for terrain in cfg.sub_terrains.values():
    terrain = copy.deepcopy(terrain)
    terrain.size = (4.0, 4.0)
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = terrain.function(0.0, spec, np.random.default_rng(7))
    assert output.geometries
    assert all(geometry.hfield is None for geometry in output.geometries)


def test_rlmpc_g23_terrain_factory_returns_independent_configs():
  first = make_rlmpc_g23_terrain_cfg()
  second = make_rlmpc_g23_terrain_cfg()

  first.sub_terrains["random_rough"].proportion = 0.0
  assert second.sub_terrains["random_rough"].proportion == 0.40
  assert RLMPC_G23_TERRAINS_CFG.sub_terrains["random_rough"].proportion == 0.40


def test_rlmpc_g23_terrain_validator_rejects_stair_name_and_type():
  stair_name_cfg = make_rlmpc_g23_terrain_cfg()
  stair_name_cfg.sub_terrains["training_stairs"] = stair_name_cfg.sub_terrains.pop(
    "low_wave"
  )
  with pytest.raises(ValueError, match="Stair terrain is not allowed"):
    validate_rlmpc_g23_terrain_cfg(stair_name_cfg)

  stair_type_cfg = make_rlmpc_g23_terrain_cfg()
  stair_type_cfg.sub_terrains["low_wave"] = BoxPyramidStairsTerrainCfg(
    proportion=0.20,
    step_height_range=(0.01, 0.08),
    step_width=0.30,
  )
  with pytest.raises(ValueError, match="Stair terrain is not allowed"):
    validate_rlmpc_g23_terrain_cfg(stair_type_cfg)
