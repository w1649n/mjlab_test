"""Configuration and state-contract tests for the G23 RL-MPC task."""

import numpy as np
import torch

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.rl_mpc.config.g23.env_cfgs import (
  syncai_g23_pure_mpc_validation_env_cfg,
)
from mjlab.tasks.rl_mpc.controller.state_adapter import pack_mpc_controller_batch
from mjlab.tasks.rl_mpc.mdp.actions import MpcFootPlacementActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK_ID = "Mjlab-RLMPC-Flat-SyncAI-G23"


def test_rl_mpc_task_is_registered_with_foot_placement_only() -> None:
  assert TASK_ID in list_tasks()
  cfg = load_env_cfg(TASK_ID)

  assert set(cfg.actions) == {"foot_placement"}
  action = cfg.actions["foot_placement"]
  assert isinstance(action, MpcFootPlacementActionCfg)
  assert action.foot_offset_scale_xy == (0.08, 0.03)
  assert action.formulation == "syncai"
  assert action.controller_dt == 0.01
  assert action.iterations_between_mpc == 2
  assert action.gait_period == 0.5
  assert action.body_mass == 15.7
  assert action.foot_landing_height == 0.019
  assert action.contact_sensor_name == "feet_ground_contact"
  assert action.terrain_sensor_name is None
  assert action.enable_stand_mode
  assert action.symmetric_residuals
  assert action.stop_reposition_steps == 2
  assert cfg.rewards["straight_foot_placement_symmetry"].weight == -0.1
  assert action.stand_enter_linear_speed == 0.03
  assert action.stand_exit_linear_speed == 0.08
  assert action.stand_command_hold_time == 0.20
  assert action.stand_contact_hold_time == 0.05
  assert action.stand_contact_loss_time == 0.05
  assert action.stand_arm_timeout == 0.20
  assert action.stand_max_vertical_speed == 0.10
  assert action.stand_max_roll_pitch_rate == 0.20
  assert action.stand_foot_search_rate == 0.05
  assert action.stand_foot_search_depth == 0.08
  assert action.stand_mpc_xy_position_weight == 5.0
  assert action.stand_mpc_vxy_weight == 3.0
  assert action.stand_kp_cartesian == (80.0, 80.0, 30.0)
  assert action.stand_kd_cartesian == (8.0, 8.0, 8.0)
  assert action.stand_hipx_kp == 12.0
  assert action.stand_hipx_kd == 1.0
  assert cfg.rewards["foot_placement_offset_l2"].weight == -0.02
  lateral_reward = cfg.rewards["foot_placement_lateral_offset_l2"]
  assert lateral_reward.func.__name__ == "foot_placement_lateral_offset_l2"
  assert lateral_reward.weight == -0.05
  assert "mean_abs_lateral_foot_placement_action" in cfg.metrics
  hip_metric = cfg.metrics["max_abs_hip_x"]
  assert hip_metric.func.__name__ == "max_abs_hip_x"
  assert hip_metric.reduce == "max"
  assert "max_terrain_scan_miss_fraction" not in cfg.metrics
  assert "terrain_scan_failed" not in cfg.terminations
  assert cfg.rewards["stopped_joint_pose_l2"].weight == -1.0
  assert not {"terrain_scan", "foot_height_scan", "base_height_scan"} & {
    sensor.name for sensor in cfg.scene.sensors
  }
  for group in cfg.observations.values():
    assert "height_scan" not in group.terms
    assert "foot_height" not in group.terms
    assert set(group.terms) == {"state_history"}
    assert group.terms["state_history"].func.__name__ == "mpc_state_history"

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert twist.ranges.lin_vel_x == (-0.2, 0.5)
  assert twist.ranges.lin_vel_y == (-0.15, 0.15)
  assert twist.ranges.ang_vel_z == (-0.5, 0.5)
  assert twist.sample_single_axis_commands
  assert twist.rel_standing_envs == 0.25

  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.max_init_terrain_level == 0


def test_rl_mpc_play_and_validation_modes_disable_training_perturbations() -> None:
  play = load_env_cfg(TASK_ID, play=True)
  validation = syncai_g23_pure_mpc_validation_env_cfg()

  assert play.episode_length_s >= 1e9
  assert not play.observations["actor"].enable_corruption
  assert play.curriculum == {}
  assert play.scene.terrain is not None
  assert play.scene.terrain.max_init_terrain_level == 5
  play_action = play.actions["foot_placement"]
  assert isinstance(play_action, MpcFootPlacementActionCfg)
  assert play_action.terrain_sensor_name is None

  assert validation.scene.num_envs == 3
  assert validation.curriculum == {}
  assert not validation.observations["actor"].enable_corruption
  validation_action = validation.actions["foot_placement"]
  assert isinstance(validation_action, MpcFootPlacementActionCfg)
  assert validation_action.terrain_sensor_name is None
  for cfg in (play, validation):
    assert "push_robot" not in cfg.events
    assert "encoder_bias" not in cfg.events
    assert "base_com" not in cfg.events


def test_rl_mpc_ppo_contract() -> None:
  rl_cfg = load_rl_cfg(TASK_ID)
  assert isinstance(rl_cfg, RslRlOnPolicyRunnerCfg)
  assert rl_cfg.clip_actions == 1.0
  assert rl_cfg.actor.hidden_dims == (512, 256, 128)
  assert rl_cfg.critic.hidden_dims == (512, 256, 128)
  assert rl_cfg.actor.distribution_cfg is not None
  assert rl_cfg.actor.distribution_cfg["init_std"] == 0.25
  assert rl_cfg.actor.distribution_cfg["std_range"] == (0.05, 0.50)
  assert rl_cfg.algorithm.entropy_coef == 0.001
  assert rl_cfg.num_steps_per_env == 24
  assert rl_cfg.max_iterations == 5_000


def test_state_adapter_reorders_quaternion_and_removes_environment_origin() -> None:
  joint_pos = torch.arange(24, dtype=torch.float32).view(2, 12)
  joint_vel = joint_pos + 100.0
  root_pos = torch.tensor([[11.0, 22.0, 3.5], [-4.0, 7.0, 2.0]])
  env_origins = torch.tensor([[10.0, 20.0, 3.0], [-5.0, 5.0, 1.5]])
  quat_wxyz = torch.tensor([[1.0, 0.1, 0.2, 0.3], [0.5, 0.4, 0.3, 0.2]])
  lin_vel = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  ang_vel = -lin_vel
  commands = torch.tensor([[0.2, 0.0, 0.1], [-0.2, 0.1, -0.1]])
  offsets = torch.arange(16, dtype=torch.float32).view(2, 4, 2) * 0.001
  contacts = torch.tensor([[True, False, True, False], [True, True, True, True]])
  terrain_samples = torch.tensor(
    [
      [[10.0, 20.0, 3.0], [10.1, 20.0, 3.02]],
      [[-5.0, 5.0, 1.5], [-4.9, 5.0, 1.54]],
    ]
  )
  local_terrain_samples = terrain_samples - env_origins[:, None, :]

  batch = pack_mpc_controller_batch(
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    root_pos_w=root_pos,
    root_quat_wxyz=quat_wxyz,
    root_lin_vel_w=lin_vel,
    root_ang_vel_w=ang_vel,
    env_origins=env_origins,
    commands=commands,
    foot_placement_offsets=offsets,
    foot_contacts=contacts,
    terrain_height_samples=local_terrain_samples,
  )

  np.testing.assert_allclose(batch.dof_states[0, :, 0], joint_pos[0].numpy())
  np.testing.assert_allclose(batch.dof_states[0, :, 1], joint_vel[0].numpy())
  np.testing.assert_allclose(
    batch.body_states[:, 0:3], (root_pos - env_origins).numpy()
  )
  np.testing.assert_allclose(
    batch.body_states[:, 3:7], quat_wxyz[:, [1, 2, 3, 0]].numpy()
  )
  np.testing.assert_allclose(batch.commands, commands.numpy())
  np.testing.assert_allclose(batch.foot_placement_offsets, offsets.numpy())
  np.testing.assert_array_equal(batch.foot_contacts, contacts.numpy())
  assert batch.terrain_height_samples is not None
  np.testing.assert_allclose(
    batch.terrain_height_samples, local_terrain_samples.numpy()
  )
