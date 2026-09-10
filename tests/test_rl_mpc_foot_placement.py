"""Unit tests for the RL-MPC foot-placement residual action contract."""

import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from mjlab.tasks.rl_mpc.controller.foot_placement import (
  FOOT_PLACEMENT_ACTION_DIM,
  FOOT_PLACEMENT_ACTION_NAMES,
  FOOT_PLACEMENT_HARD_LIMIT_M,
  LEG_ORDER,
  FootPlacementOffsetCfg,
  FootPlacementOffsetLatch,
  yaw_aligned_to_world,
)
from mjlab.tasks.rl_mpc.controller.observation_history import MpcObservationHistory
from mjlab.tasks.rl_mpc.mdp.actions import (
  MpcFootPlacementAction,
  MpcFootPlacementActionCfg,
  extract_terrain_height_samples,
)
from mjlab.tasks.rl_mpc.mdp.metrics import terrain_scan_miss_fraction
from mjlab.tasks.rl_mpc.mdp.observations import mpc_gait_mode
from mjlab.tasks.rl_mpc.mdp.rewards import foot_placement_lateral_offset_l2
from mjlab.tasks.rl_mpc.mdp.terminations import terrain_scan_failed


def test_foot_placement_action_layout_is_leg_major() -> None:
  assert LEG_ORDER == ("FL", "FR", "HL", "HR")
  assert FOOT_PLACEMENT_ACTION_DIM == 8
  assert FOOT_PLACEMENT_ACTION_NAMES == (
    "FL_dx",
    "FL_dy",
    "FR_dx",
    "FR_dy",
    "HL_dx",
    "HL_dy",
    "HR_dx",
    "HR_dy",
  )


def test_actual_mpc_gait_mode_is_exposed_as_one_hot() -> None:
  term = SimpleNamespace(gait_modes=torch.tensor([0, 1, 2], dtype=torch.long))
  action_manager = SimpleNamespace(get_term=lambda _: term)
  env = SimpleNamespace(num_envs=3, action_manager=action_manager)

  observation = mpc_gait_mode(env)  # type: ignore[arg-type]

  torch.testing.assert_close(observation, torch.eye(3))


def test_actions_are_scaled_and_hard_clipped_in_metres() -> None:
  mapper = FootPlacementOffsetLatch(
    num_envs=1,
    device="cpu",
    cfg=FootPlacementOffsetCfg(scale_xy=(0.04, 0.16)),
  )
  actions = torch.tensor([[0.5, 0.5, 2.0, -2.0, -1.0, 1.0, 0.0, -0.25]])

  offsets = mapper.process_actions(actions)

  torch.testing.assert_close(
    offsets,
    torch.tensor(
      [
        [
          [0.02, FOOT_PLACEMENT_HARD_LIMIT_M],
          [0.04, -FOOT_PLACEMENT_HARD_LIMIT_M],
          [-0.04, FOOT_PLACEMENT_HARD_LIMIT_M],
          [0.0, -0.04],
        ]
      ]
    ),
  )


def test_lateral_offset_cost_uses_only_dy_actions() -> None:
  action = torch.tensor(
    [
      [0.9, 0.1, -0.8, -0.2, 0.7, 0.3, -0.6, -0.4],
      [4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0],
    ]
  )
  env = SimpleNamespace(action_manager=SimpleNamespace(action=action))

  cost = foot_placement_lateral_offset_l2(env)  # type: ignore[arg-type]

  torch.testing.assert_close(cost, torch.tensor([0.30, 0.0]))


def test_task_scale_caps_lateral_residual_at_three_centimetres() -> None:
  mapper = FootPlacementOffsetLatch(
    num_envs=1,
    device="cpu",
    cfg=FootPlacementOffsetCfg(scale_xy=(0.08, 0.03)),
  )

  offsets = mapper.process_actions(torch.tensor([[2.0, 2.0] * 4]))

  torch.testing.assert_close(offsets[..., 0], torch.full((1, 4), 0.08))
  torch.testing.assert_close(offsets[..., 1], torch.full((1, 4), 0.03))


def test_offsets_latch_independently_only_at_swing_entry() -> None:
  latch = FootPlacementOffsetLatch(num_envs=1, device="cpu")
  first = torch.tensor([[1.0, 0.5, 0.8, 0.4, 0.6, 0.3, 0.4, 0.2]])
  second = -first
  latch.process_actions(first)

  all_stance = torch.zeros((1, 4), dtype=torch.bool)
  latch.latch_on_swing_entry(all_stance)
  torch.testing.assert_close(latch.latched_offsets, torch.zeros((1, 4, 2)))

  fl_hr_swing = torch.tensor([[True, False, False, True]])
  latch.latch_on_swing_entry(fl_hr_swing)
  first_offsets = first.view(1, 4, 2) * 0.08
  torch.testing.assert_close(latch.latched_offsets[0, 0], first_offsets[0, 0])
  torch.testing.assert_close(latch.latched_offsets[0, 3], first_offsets[0, 3])

  latch.process_actions(second)
  latch.latch_on_swing_entry(fl_hr_swing)
  torch.testing.assert_close(latch.latched_offsets[0, 0], first_offsets[0, 0])
  torch.testing.assert_close(latch.latched_offsets[0, 3], first_offsets[0, 3])

  fr_swing = torch.tensor([[False, True, False, False]])
  latch.latch_on_swing_entry(fr_swing)
  second_offsets = second.view(1, 4, 2) * 0.08
  torch.testing.assert_close(latch.latched_offsets[0, 1], second_offsets[0, 1])
  torch.testing.assert_close(latch.latched_offsets[0, 0], first_offsets[0, 0])


def test_yaw_aligned_offsets_rotate_into_world_frame() -> None:
  offsets = torch.zeros((1, 4, 2))
  offsets[:, :, 0] = 0.08

  world = yaw_aligned_to_world(offsets, torch.tensor([math.pi / 2]))

  torch.testing.assert_close(world[..., 0], torch.zeros((1, 4)), atol=1e-6, rtol=0)
  torch.testing.assert_close(world[..., 1], torch.full((1, 4), 0.08))


def test_reset_clears_only_selected_environments_and_edge_state() -> None:
  latch = FootPlacementOffsetLatch(num_envs=2, device="cpu")
  latch.process_actions(torch.ones((2, FOOT_PLACEMENT_ACTION_DIM)))
  latch.latch_on_swing_entry(torch.ones((2, 4), dtype=torch.bool))

  latch.reset(torch.tensor([0]))

  torch.testing.assert_close(latch.raw_actions[0], torch.zeros(8))
  torch.testing.assert_close(latch.pending_offsets[0], torch.zeros((4, 2)))
  torch.testing.assert_close(latch.latched_offsets[0], torch.zeros((4, 2)))
  torch.testing.assert_close(latch.latched_offsets[1], torch.full((4, 2), 0.08))

  latch.process_actions(torch.zeros((2, FOOT_PLACEMENT_ACTION_DIM)))
  latch.latch_on_swing_entry(torch.ones((2, 4), dtype=torch.bool))
  torch.testing.assert_close(latch.latched_offsets[0], torch.zeros((4, 2)))
  torch.testing.assert_close(latch.latched_offsets[1], torch.full((4, 2), 0.08))


@pytest.mark.parametrize(
  "scale_xy",
  [(-0.01, 0.08), (float("nan"), 0.08), (float("inf"), 0.08)],
)
def test_invalid_scale_is_rejected(scale_xy: tuple[float, float]) -> None:
  with pytest.raises(ValueError):
    FootPlacementOffsetCfg(scale_xy=scale_xy)


def test_invalid_action_shape_is_rejected() -> None:
  latch = FootPlacementOffsetLatch(num_envs=2, device="cpu")
  with pytest.raises(ValueError, match="actions must have shape"):
    latch.process_actions(torch.zeros((2, 12)))


def test_terrain_scan_defaults_to_strict_error_policy() -> None:
  cfg = MpcFootPlacementActionCfg(entity_name="robot")

  assert cfg.terrain_scan_miss_policy == "error"
  assert cfg.terrain_scan_min_valid_fraction == 0.9


def test_terrain_scan_miss_keeps_query_xy_and_uses_same_env_nearest_z() -> None:
  env_origins = torch.tensor([[10.0, 20.0, 1.0], [-10.0, -20.0, 2.0]])
  local_samples = torch.tensor(
    [
      [[0.0, 0.0, 0.10], [0.1, 0.0, 0.80], [1.0, 0.0, 0.90], [0.0, 1.0, 0.40]],
      [[0.0, 0.0, 1.10], [0.0, 0.1, 1.80], [0.0, 1.0, 1.90], [1.0, 0.0, 1.40]],
    ]
  )
  hit_pos_w = local_samples + env_origins[:, None, :]
  distances = torch.ones((2, 4))
  distances[:, 1] = -1.0

  result = extract_terrain_height_samples(
    hit_pos_w=hit_pos_w,
    distances=distances,
    env_origins=env_origins,
    min_valid_fraction=0.75,
  )

  torch.testing.assert_close(result.samples[:, 1, :2], local_samples[:, 1, :2])
  torch.testing.assert_close(result.samples[:, 1, 2], torch.tensor([0.10, 1.10]))
  torch.testing.assert_close(result.miss_fraction, torch.tensor([0.25, 0.25]))
  torch.testing.assert_close(result.miss_count, torch.tensor([1, 1]))
  assert not torch.any(result.below_min_valid_fraction)


def test_terrain_scan_marks_too_few_valid_rays_and_keeps_samples_finite() -> None:
  result = extract_terrain_height_samples(
    hit_pos_w=torch.tensor([[[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]]),
    distances=torch.full((1, 2), -1.0),
    env_origins=torch.tensor([[0.0, 0.0, 1.0]]),
    min_valid_fraction=0.9,
  )

  assert result.below_min_valid_fraction.tolist() == [True]
  assert result.miss_count.tolist() == [2]
  torch.testing.assert_close(
    result.samples[0, :, :2], torch.tensor([[0.0, 0.0], [0.1, 0.0]])
  )
  torch.testing.assert_close(result.samples[0, :, 2], torch.zeros(2))
  assert torch.all(torch.isfinite(result.samples))


@pytest.mark.parametrize(
  ("hit_pos_w", "distances", "env_origins", "error"),
  [
    (torch.zeros((1, 2)), torch.ones((1, 2)), torch.zeros((1, 3)), "hit_pos_w"),
    (torch.zeros((1, 2, 3)), torch.ones((1, 3)), torch.zeros((1, 3)), "distances"),
    (torch.zeros((1, 2, 3)), torch.ones((1, 2)), torch.zeros((2, 3)), "env_origins"),
  ],
)
def test_terrain_scan_rejects_shape_mismatch(
  hit_pos_w: torch.Tensor,
  distances: torch.Tensor,
  env_origins: torch.Tensor,
  error: str,
) -> None:
  with pytest.raises(ValueError, match=error):
    extract_terrain_height_samples(
      hit_pos_w=hit_pos_w,
      distances=distances,
      env_origins=env_origins,
      min_valid_fraction=0.9,
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_terrain_scan_rejects_nonfinite_input(invalid: float) -> None:
  hit_pos_w = torch.zeros((1, 2, 3))
  hit_pos_w[0, 1, 2] = invalid

  with pytest.raises(ValueError, match="NaN or Inf"):
    extract_terrain_height_samples(
      hit_pos_w=hit_pos_w,
      distances=torch.ones((1, 2)),
      env_origins=torch.zeros((1, 3)),
      min_valid_fraction=0.9,
    )


def test_terrain_scan_metric_and_termination_are_per_environment() -> None:
  action = SimpleNamespace(
    terrain_scan_miss_fraction=torch.tensor([0.0, 0.25]),
    terrain_scan_failed=torch.tensor([False, True]),
  )
  env = SimpleNamespace(
    num_envs=2,
    action_manager=SimpleNamespace(get_term=lambda _: action),
  )

  torch.testing.assert_close(
    terrain_scan_miss_fraction(env),  # type: ignore[arg-type]
    torch.tensor([0.0, 0.25]),
  )
  torch.testing.assert_close(
    terrain_scan_failed(env),  # type: ignore[arg-type]
    torch.tensor([False, True]),
  )


def test_action_reset_clears_only_selected_terrain_scan_status() -> None:
  action = cast(Any, object.__new__(MpcFootPlacementAction))
  action._env = SimpleNamespace(num_envs=2, device="cpu")
  action._offsets = SimpleNamespace(reset=lambda _env_ids: None)
  action._observation_history = MpcObservationHistory(2, "cpu")
  action._mpc_snapshot = torch.zeros((2, 24))
  action._action_history = torch.ones((2, 4, 8))
  action._torques = torch.ones((2, 12))
  action._terrain_scan_miss_fraction = torch.tensor([0.1, 0.2])
  action._terrain_scan_failed = torch.tensor([True, True])
  action._terrain_scan_missed_ray_count = 7
  action._terrain_scan_failure_count = 2
  action._controllers = [
    SimpleNamespace(reset=lambda: None, gait_mode=2, stand_ready=True),
    SimpleNamespace(reset=lambda: None, gait_mode=2, stand_ready=True),
  ]
  action._gait_modes = torch.zeros(2, dtype=torch.long)
  action._stand_ready = torch.zeros(2, dtype=torch.bool)

  action.reset(torch.tensor([1]))

  assert action.terrain_scan_failed.tolist() == [True, False]
  torch.testing.assert_close(
    action.terrain_scan_miss_fraction, torch.tensor([0.1, 0.0])
  )
  torch.testing.assert_close(action.torques[0], torch.ones(12))
  torch.testing.assert_close(action.torques[1], torch.zeros(12))
  assert action.terrain_scan_missed_ray_count == 7
  assert action.terrain_scan_failure_count == 2


class _CountingController:
  observation_snapshot = np.zeros(24, dtype=np.float32)
  gait_mode = 0
  stand_ready = False

  def __init__(self) -> None:
    self.step_count = 0

  def step(self, *_args: object) -> np.ndarray:
    self.step_count += 1
    return torch.ones(12).numpy()


def _make_terrain_scan_action(
  miss_policy: str = "nearest_valid",
) -> tuple[Any, list[_CountingController]]:
  action = cast(Any, object.__new__(MpcFootPlacementAction))
  scene = SimpleNamespace(env_origins=torch.zeros((2, 3)))
  action._env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    scene=scene,
    command_manager=SimpleNamespace(get_command=lambda _name: torch.zeros((2, 3))),
  )
  action.cfg = SimpleNamespace(
    command_name="twist",
    contact_sensor_name="feet_ground_contact",
    terrain_sensor_name="terrain_scan",
    terrain_scan_miss_policy=miss_policy,
    terrain_scan_min_valid_fraction=0.9,
    fail_on_controller_error=True,
  )
  action._offsets = SimpleNamespace(
    process_actions=lambda _actions: torch.zeros((2, 4, 2)),
    pending_offsets=torch.zeros((2, 4, 2)),
  )
  action._contact_sensor = SimpleNamespace(
    data=SimpleNamespace(found=torch.ones((2, 4)))
  )
  distances = torch.ones((2, 10))
  distances[1, -2:] = -1.0
  action._terrain_sensor = SimpleNamespace(
    data=SimpleNamespace(
      hit_pos_w=torch.zeros((2, 10, 3)),
      distances=distances,
    )
  )
  action._observation_history = MpcObservationHistory(2, "cpu")
  action._mpc_snapshot = torch.zeros((2, 24))
  action._action_history = torch.zeros((2, 4, 8))
  action._controller_substeps = 2
  action._joint_ids = torch.arange(12)
  action._torque_limits = torch.full((1, 12), 100.0)
  action._torques = torch.full((2, 12), 7.0)
  action._terrain_scan_miss_fraction = torch.zeros(2)
  action._terrain_scan_failed = torch.zeros(2, dtype=torch.bool)
  action._terrain_scan_initialized = True
  action._terrain_scan_missed_ray_count = 0
  action._terrain_scan_failure_count = 0
  action.controller_error_count = 0
  controllers = [_CountingController(), _CountingController()]
  action._controllers = controllers
  action._gait_modes = torch.zeros(2, dtype=torch.long)
  action._stand_ready = torch.zeros(2, dtype=torch.bool)
  action._entity = SimpleNamespace(
    data=SimpleNamespace(
      joint_pos=torch.zeros((2, 12)),
      joint_vel=torch.zeros((2, 12)),
      root_link_pos_w=torch.zeros((2, 3)),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
      root_link_lin_vel_w=torch.zeros((2, 3)),
      root_link_ang_vel_w=torch.zeros((2, 3)),
    )
  )
  return action, controllers


def test_tolerant_severe_scan_skips_only_failed_environment_controller() -> None:
  action, controllers = _make_terrain_scan_action()

  action.process_actions(torch.zeros((2, FOOT_PLACEMENT_ACTION_DIM)))

  assert [controller.step_count for controller in controllers] == [1, 0]
  torch.testing.assert_close(action.torques[0], torch.ones(12))
  torch.testing.assert_close(action.torques[1], torch.zeros(12))
  assert action.terrain_scan_failed.tolist() == [False, True]
  torch.testing.assert_close(
    action.terrain_scan_miss_fraction, torch.tensor([0.0, 0.2])
  )
  assert action.terrain_scan_missed_ray_count == 2
  assert action.terrain_scan_failure_count == 1


def test_tolerant_severe_initial_scan_fails_fast() -> None:
  action, controllers = _make_terrain_scan_action()
  action._terrain_scan_initialized = False

  with pytest.raises(RuntimeError, match="severely incomplete during initialization"):
    action.process_actions(torch.zeros((2, FOOT_PLACEMENT_ACTION_DIM)))

  assert [controller.step_count for controller in controllers] == [0, 0]
  assert action.terrain_scan_failure_count == 1


def test_tolerant_multiple_severe_scans_fail_fast() -> None:
  action, controllers = _make_terrain_scan_action()
  action._terrain_sensor.data.distances[:, -2:] = -1.0

  with pytest.raises(RuntimeError, match="in multiple environments"):
    action.process_actions(torch.zeros((2, FOOT_PLACEMENT_ACTION_DIM)))

  assert [controller.step_count for controller in controllers] == [0, 0]
  assert action.terrain_scan_failure_count == 2


def test_tolerant_single_seam_ray_miss_keeps_all_controllers_running() -> None:
  action, controllers = _make_terrain_scan_action()
  action._terrain_sensor.data.hit_pos_w = torch.zeros((2, 187, 3))
  action._terrain_sensor.data.distances = torch.ones((2, 187))
  action._terrain_sensor.data.distances[1, 151] = -1.0

  action.process_actions(torch.zeros((2, FOOT_PLACEMENT_ACTION_DIM)))

  assert [controller.step_count for controller in controllers] == [1, 1]
  assert not torch.any(action.terrain_scan_failed)
  torch.testing.assert_close(
    action.terrain_scan_miss_fraction,
    torch.tensor([0.0, 1.0 / 187.0]),
  )
  assert action.terrain_scan_missed_ray_count == 1
  assert action.terrain_scan_failure_count == 0


def test_strict_scan_policy_rejects_any_missed_ray_before_controller() -> None:
  action, controllers = _make_terrain_scan_action(miss_policy="error")
  action._terrain_sensor.data.hit_pos_w = torch.zeros((2, 187, 3))
  action._terrain_sensor.data.distances = torch.ones((2, 187))
  action._terrain_sensor.data.distances[1, 151] = -1.0

  with pytest.raises(RuntimeError, match="missed 1 ground ray"):
    action.process_actions(torch.zeros((2, FOOT_PLACEMENT_ACTION_DIM)))

  assert [controller.step_count for controller in controllers] == [0, 0]
  assert action.terrain_scan_missed_ray_count == 1
  assert action.terrain_scan_failure_count == 1


def test_blind_policy_history_and_controller_cadence() -> None:
  action, controllers = _make_terrain_scan_action()
  action._terrain_sensor = None
  action._entity.set_joint_effort_target = lambda *args, **kwargs: None
  # Five 20 ms policy steps: each holds its action over two 10 ms controls.
  for i in range(5):
    action.process_actions(torch.full((2, 8), i * 0.4))
    for _ in range(4):
      action.apply_actions()
  assert [controller.step_count for controller in controllers] == [10, 10]
  expected = torch.tensor([1.0, 1.0, 0.8, 0.4]).repeat_interleave(8)
  torch.testing.assert_close(action.action_history[0], expected)
  for controller in controllers:
    controller.reset = lambda: None
  action._offsets.reset = lambda _: None
  action.reset(torch.tensor([1]))
  torch.testing.assert_close(action.action_history[0], expected)
  torch.testing.assert_close(action.action_history[1], torch.zeros(32))


def test_stopped_pose_penalty_is_command_and_mode_gated() -> None:
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.tasks.rl_mpc.mdp.rewards import stopped_joint_pose_l2

  target = torch.tensor([0.0, -0.8, 1.6] * 4).repeat(5, 1)
  current = target.clone()
  current[1:, 0] += 0.2
  commands = torch.zeros((5, 3))
  commands[4, 0] = 0.2
  env = SimpleNamespace(
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(joint_pos=current, default_joint_pos=target)
      )
    },
    command_manager=SimpleNamespace(get_command=lambda _: commands),
    action_manager=SimpleNamespace(
      get_term=lambda _: SimpleNamespace(gait_modes=torch.tensor([2, 2, 1, 0, 2]))
    ),
  )
  reward = stopped_joint_pose_l2(env, SceneEntityCfg("robot"))
  torch.testing.assert_close(reward, torch.tensor([0.0, 0.04, 0.04, 0.0, 0.0]))


def test_symmetry_reward_bounds_turning_but_leaves_stopping_unconstrained() -> None:
  from mjlab.tasks.rl_mpc.mdp.rewards import straight_foot_placement_symmetry

  actions = torch.tensor([0.2, 0.1, 0.2, -0.1, 0.3, 0.1, 0.3, -0.1]).repeat(5, 1)
  actions[1:, 6] += 0.5
  commands = torch.tensor(
    [[0.2, 0, 0], [0.2, 0, 0], [0, 0.1, 0], [0.2, 0, 0.2], [0, 0, 0]]
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_command=lambda _: commands),
    action_manager=SimpleNamespace(
      action=actions,
      get_term=lambda _: SimpleNamespace(gait_modes=torch.tensor([0, 0, 0, 0, 1])),
    ),
  )
  torch.testing.assert_close(
    straight_foot_placement_symmetry(env),
    torch.tensor([0, 0.25, 0.57953125, 0.57953125, 0]),
  )


def test_complete_state_frame_order_and_shared_history_cache() -> None:
  action, controllers = _make_terrain_scan_action()
  action._terrain_sensor = None
  action._env.common_step_counter = 0
  command = torch.tensor([[0.2, 0.1, 0.3]]).repeat(2, 1)
  action._env.command_manager.get_command = lambda _: command
  data = action._entity.data
  data.root_link_quat_w[:] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
  data.root_link_lin_vel_w[:] = torch.tensor([1.0, 2.0, 3.0])
  data.root_link_ang_vel_w[:] = torch.tensor([4.0, 5.0, 6.0])
  data.joint_pos[:] = torch.arange(12)
  data.joint_vel[:] = torch.arange(12) + 20
  snapshot = torch.cat(
    (torch.arange(8), torch.tensor([1, 0, 0, 1]), torch.arange(12) + 12)
  ).float()
  action._mpc_snapshot[:] = snapshot
  action._offsets.pending_offsets[:] = torch.tensor([0.04, -0.015])
  action._gait_modes[:] = 1
  history = action.state_history.clone().reshape(2, 5, 75)
  frame = history[0, 0]
  torch.testing.assert_close(frame[0:3], torch.tensor([0.0, 0.0, math.pi / 2]))
  torch.testing.assert_close(frame[3:9], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
  torch.testing.assert_close(frame[9:21], torch.arange(12).float())
  torch.testing.assert_close(frame[21:33], torch.arange(12).float() + 20)
  torch.testing.assert_close(frame[33:41], torch.arange(8).float())
  torch.testing.assert_close(frame[41:45], torch.ones(4))
  torch.testing.assert_close(frame[45:49], torch.tensor([1.0, 0.0, 0.0, 1.0]))
  torch.testing.assert_close(frame[49:61], (torch.arange(12).float() + 12) / 12)
  torch.testing.assert_close(frame[61:64], torch.tensor([-0.1, 0.2, 0.3]))
  torch.testing.assert_close(frame[64:72], torch.tensor([0.04, -0.015] * 4))
  torch.testing.assert_close(frame[72:75], torch.tensor([0.0, 1.0, 0.0]))
  for age in range(5):
    torch.testing.assert_close(history[0, age], frame)
  data.joint_pos[:] += 100
  # Actor and critic may request the observation separately: no double advance.
  torch.testing.assert_close(action.state_history.reshape(2, 5, 75), history)
  action._env.common_step_counter += 1
  newer = action.state_history.clone().reshape(2, 5, 75)
  torch.testing.assert_close(newer[0, 0, 9:21], torch.arange(12).float() + 100)
  torch.testing.assert_close(newer[0, 1], frame)
  action._offsets.reset = lambda _: None
  for controller in controllers:
    controller.reset = lambda: None
  action.reset(torch.tensor([1]))
  reset = action.state_history.reshape(2, 5, 75)
  torch.testing.assert_close(reset[0], newer[0])
  for age in range(5):
    torch.testing.assert_close(reset[1, age], reset[1, 0])


def test_whole_history_retains_exactly_five_frames() -> None:
  history = MpcObservationHistory(2, "cpu")
  for step in range(8):
    output = history.update(torch.full((2, 75), float(step)), step)
  torch.testing.assert_close(
    output[0].reshape(5, 75)[:, 0], torch.tensor([7.0, 6.0, 5.0, 4.0, 3.0])
  )
  with pytest.raises(ValueError, match="Non-finite"):
    history.update(torch.full((2, 75), float("nan")), 8)
