"""Tests for the hospital navigation task."""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.navigation import assets
from mjlab.tasks.navigation.mdp.geometry import obstacle_distance
from mjlab.tasks.navigation.mdp.hsfm import HsfmParams, hsfm_step
from mjlab.tasks.navigation.mdp.lidar import PlanarLidarPatternCfg
from mjlab.tasks.registry import load_env_cfg

_NO_BOXES = torch.zeros(0, 4)


def _simulate(state, goal, active, steps=400, boxes=_NO_BOXES, **kwargs):
  for _ in range(steps):
    state = hsfm_step(state, goal, active, boxes, HsfmParams(), dt=0.025, **kwargs)
  return state


def test_hsfm_pedestrian_walks_to_goal():
  state = torch.zeros(1, 1, 6)
  goal = torch.tensor([[[5.0, 0.0]]])
  active = torch.ones(1, 1, dtype=torch.bool)
  state = _simulate(state, goal, active, steps=200)
  assert 2.0 < state[0, 0, 0] < 5.5
  assert torch.linalg.norm(state[0, 0, 2:4]) <= HsfmParams().max_speed + 1e-5


def test_hsfm_pedestrians_do_not_overlap_or_move_when_inactive():
  state = torch.zeros(1, 3, 6)
  state[0, 1, 0] = 6.0
  state[0, 1, 4] = torch.pi
  state[0, 2, 0:2] = torch.tensor([3.0, 0.1])
  goal = torch.tensor([[[6.0, 0.0], [0.0, 0.0], [9.0, 9.0]]])
  active = torch.tensor([[True, True, False]])
  parked = state[0, 2].clone()
  min_gap = torch.inf
  for _ in range(400):
    state = _simulate(state, goal, active, steps=1)
    min_gap = min(min_gap, torch.linalg.norm(state[0, 0, :2] - state[0, 1, :2]).item())
  assert min_gap > 1.5 * HsfmParams().radius
  assert torch.equal(state[0, 2], parked)


def test_hsfm_pedestrian_stays_out_of_obstacles_and_avoids_robot():
  # A wall across the straight path to the goal.
  boxes = torch.tensor([[2.0, -1.0, 2.5, 1.0]])
  state = torch.zeros(1, 1, 6)
  goal = torch.tensor([[[5.0, 0.3]]])
  active = torch.ones(1, 1, dtype=torch.bool)
  for _ in range(600):
    state = _simulate(state, goal, active, steps=1, boxes=boxes)
    x, y = state[0, 0, :2].tolist()
    assert not (2.0 < x < 2.5 and -1.0 < y < 1.0)

  # A robot standing on the path pushes the pedestrian sideways.
  state = torch.zeros(1, 1, 6)
  state[0, 0, 1] = 0.05
  robot = {"robot_pos": torch.tensor([[2.0, 0.0]]), "robot_vel": torch.zeros(1, 2)}
  deflection = 0.0
  for _ in range(200):
    state = _simulate(state, goal, active, steps=1, robot_radius=0.2, **robot)
    deflection = max(deflection, abs(state[0, 0, 1].item()))
  assert deflection > 0.3


def test_planar_lidar_pattern():
  offsets, directions = PlanarLidarPatternCfg(num_rays=4).generate_rays(None, "cpu")
  expected = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], [0, -1.0, 0]])
  assert torch.allclose(directions, expected, atol=1e-6)
  assert torch.count_nonzero(offsets) == 0


@pytest.mark.parametrize(
  "task_id", ["Mjlab-Navigation-Hospital", "Mjlab-Navigation-Hospital-Humans"]
)
def test_navigation_env_steps(task_id):
  device = get_test_device()
  cfg = load_env_cfg(task_id)
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    obs, _ = env.reset()
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape == (16, 2 * (2 + 108))
    humans_name = "humans" if "humans" in cfg.scene.entities else None

    # The lidar sees what the analytic clearance sees, up to its resolution.
    lidar = env.scene["lidar"].data.distances
    lidar = torch.where(lidar < 0, torch.inf, lidar).min(dim=-1).values
    clearance = obstacle_distance(env, humans_name=humans_name)
    assert torch.all(lidar >= clearance - 1e-3)
    assert torch.all(lidar - clearance < 0.25)

    # Driving straight ahead ends every episode in a collision or at the goal.
    forward = torch.tensor([[1.0, 0.0]], device=device).repeat(16, 1)
    done = torch.zeros(16, dtype=torch.bool, device=device)
    for _ in range(200):
      obs, reward, terminated, _, _ = env.step(forward)
      actor_obs = obs["actor"]
      assert isinstance(actor_obs, torch.Tensor)
      assert torch.isfinite(actor_obs).all() and torch.isfinite(reward).all()
      assert torch.all(reward[terminated].abs() > 1.0)
      done |= terminated
    assert done.all()
    assert torch.all(obstacle_distance(env, humans_name=humans_name) > 0)
  finally:
    env.close()


def test_hospital_map_matches_lidar_height():
  hospital_map = assets.load_hospital_map()
  assert hospital_map.has_geometry.sum() == 59
  assert len(hospital_map.boxes_at_height(assets.LIDAR_HEIGHT)) == 59
