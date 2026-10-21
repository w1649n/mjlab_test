"""Tests for the G23 left-right mirror augmentation."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from mjlab.sensor import GridPatternCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.g23 import symmetry

_JOINTS = [
  f"{leg}_{j}_joint"
  for leg in ("FL", "FR", "HL", "HR")
  for j in ("HipX", "HipY", "Knee")
]


def test_joint_mirror_swaps_legs_and_negates_abduction():
  mirror = symmetry.joint_mirror(_JOINTS)
  assert mirror.perm == (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
  assert mirror.sign == (-1.0, 1.0, 1.0) * 4


def test_foot_mirror_swaps_blocks_and_flips_lateral_components():
  mirror = symmetry.foot_mirror(
    ["FL_foot", "FR_foot", "HL_foot", "HR_foot"], (1.0, -1.0, 1.0)
  )
  assert mirror.perm[:6] == (3, 4, 5, 0, 1, 2)
  assert mirror.sign[:3] == (1.0, -1.0, 1.0)


def test_grid_mirror_reflects_rays_across_the_x_axis():
  offsets, _ = GridPatternCfg(size=(1.6, 1.0), resolution=0.1).generate_rays(
    None, "cpu"
  )
  mirror = symmetry.grid_mirror(offsets)
  perm = torch.tensor(mirror.perm)
  expected = offsets[:, :2] * torch.tensor([1.0, -1.0])
  torch.testing.assert_close(offsets[perm, :2], expected, atol=1e-5, rtol=0.0)
  assert torch.equal(perm[perm], torch.arange(len(perm)))


def test_tile_and_concat_cover_flattened_history():
  vector = symmetry.Mirror((0, 1, 2), (1.0, -1.0, 1.0)).tile(2)
  assert vector.perm == (0, 1, 2, 3, 4, 5)
  assert vector.sign == (1.0, -1.0, 1.0, 1.0, -1.0, 1.0)
  both = symmetry.concat_mirrors([vector, symmetry.Mirror((1, 0), (1.0, 1.0))])
  assert both.perm[-2:] == (7, 6)


def _fake_env(term_names: list[str], dims: list[tuple[int, ...]], funcs=None):
  funcs = funcs or {}
  terms = {
    name: SimpleNamespace(func=funcs.get(name), params={"sensor_name": "feet"})
    for name in term_names
  }
  env = MagicMock()
  env.unwrapped = env
  env.device = "cpu"
  env.observation_manager.active_terms = {"actor": term_names}
  env.observation_manager.group_obs_term_dim = {"actor": dims}
  env.cfg.observations = {"actor": SimpleNamespace(terms=terms)}
  env.scene.__getitem__.return_value = SimpleNamespace(joint_names=_JOINTS)
  env.action_manager.active_terms = ["joint_pos"]
  env.action_manager.get_term.return_value = SimpleNamespace(target_names=_JOINTS)
  return env


def test_mirror_appends_mirrored_history_and_actions():
  env = _fake_env(["projected_gravity", "joint_pos", "actions"], [(6,), (24,), (12,)])
  actor = torch.randn(5, 42)
  obs = TensorDict({"actor": actor}, batch_size=[5])
  actions = torch.randn(5, 12)

  aug_obs, aug_actions = symmetry.mirror_g23_velocity(env, obs, actions)
  assert aug_obs is not None and aug_actions is not None
  assert aug_obs.batch_size == torch.Size([10])
  aug_actor = cast(torch.Tensor, aug_obs["actor"])
  torch.testing.assert_close(aug_actor[:5], actor)
  mirrored = aug_actor[5:]
  # Gravity y flips in both history frames.
  torch.testing.assert_close(mirrored[:, [1, 4]], -actor[:, [1, 4]])
  # Second joint frame: FL HipX <- -FR HipX, FL HipY <- FR HipY.
  torch.testing.assert_close(mirrored[:, 6 + 12], -actor[:, 6 + 12 + 3])
  torch.testing.assert_close(mirrored[:, 6 + 13], actor[:, 6 + 13 + 3])
  torch.testing.assert_close(aug_actions[5:, 9], -actions[:, 6])

  mirrored_obs = TensorDict({"actor": mirrored}, batch_size=[5])
  twice, _ = symmetry.mirror_g23_velocity(env, mirrored_obs, None)
  assert twice is not None
  torch.testing.assert_close(cast(torch.Tensor, twice["actor"])[5:], actor)


def test_mirror_rejects_terms_it_cannot_reflect():
  env = _fake_env(["height_scan_typo"], [(3,)])
  with pytest.raises(ValueError, match="No mirror defined"):
    symmetry.mirror_g23_velocity(env, None, torch.zeros(1, 12))

  env = _fake_env(
    ["foot_contact_forces"], [(12,)], {"foot_contact_forces": mdp.foot_contact_forces}
  )
  with pytest.raises(ValueError, match="World-frame"):
    symmetry.mirror_g23_velocity(env, None, torch.zeros(1, 12))
