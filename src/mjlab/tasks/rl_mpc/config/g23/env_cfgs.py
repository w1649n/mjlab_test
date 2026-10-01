"""SyncAI G23 residual-foot-placement RL-MPC environment."""

from __future__ import annotations

import math
from typing import cast

from mjlab.asset_zoo.robots import get_g23_rl_mpc_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.rl_mpc import mdp
from mjlab.tasks.rl_mpc.mdp.actions import MpcFootPlacementActionCfg
from mjlab.tasks.velocity.config.g23.env_cfgs import syncai_g23_rough_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .terrain import make_rlmpc_g23_terrain_cfg


def _remove_domain_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep the first training stage deterministic enough to validate pure MPC."""
  for event_name in (
    "push_robot",
    "foot_friction",
    "foot_friction_slide",
    "foot_friction_spin",
    "foot_friction_roll",
    "encoder_bias",
    "base_com",
  ):
    cfg.events.pop(event_name, None)


def syncai_g23_rl_mpc_env_cfg(
  play: bool = False,
  *,
  pure_mpc_validation: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the mild-terrain G23 RL-MPC task.

  The policy controls only eight planar foot-placement residuals.  Joint-angle
  residuals are intentionally absent.  At terrain level zero every generated
  heightfield is flat; the existing distance curriculum then unlocks continuous
  uneven terrain up to an 8 cm peak-to-peak span.
  """
  # Reuse the well-tested G23 contacts, height sensors, rewards and terminations,
  # then replace every stair-specific or joint-position-specific component.
  cfg = syncai_g23_rough_env_cfg(play=False)
  cfg.scene.entities = {"robot": get_g23_rl_mpc_robot_cfg()}

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = make_rlmpc_g23_terrain_cfg()
  # Gate zero is pure flat MPC. Training progresses to level five (8 cm) only
  # after the robot demonstrates sufficient commanded travel.
  cfg.scene.terrain.max_init_terrain_level = 0

  cfg.sim.mujoco.timestep = 0.005
  cfg.decimation = 4
  cfg.actions = {
    "foot_placement": MpcFootPlacementActionCfg(
      entity_name="robot",
      formulation="syncai",
      controller_dt=0.01,
      # Controller 100 Hz, MPC 50 Hz, independent 0.5 s trot cycle.
      iterations_between_mpc=2,
      gait_period=0.5,
      flat_ground=False,
      # Match the currently loaded mjlab MJCF (15.7 kg total), rather than the
      # controller repository's older 14.362548 kg URDF.
      body_mass=15.7,
      # The controller's kinematic foot point is the center of the G23's
      # 22 mm sphere. Preserve its legacy 3 mm penetration target at the sole.
      foot_landing_height=0.019,
      contact_sensor_name="feet_ground_contact",
      # Blind controller: contact feedback only, no terrain height scan.
      terrain_sensor_name=None,
      enable_stand_mode=True,
      symmetric_residuals=True,
      stop_reposition_steps=2,
      stand_enter_linear_speed=0.03,
      stand_enter_yaw_rate=0.05,
      stand_exit_linear_speed=0.08,
      stand_exit_yaw_rate=0.10,
      stand_command_hold_time=0.20,
      stand_contact_hold_time=0.05,
      stand_contact_loss_time=0.05,
      stand_arm_timeout=0.20,
      stand_max_linear_speed=0.10,
      stand_max_vertical_speed=0.10,
      stand_max_yaw_rate=0.20,
      stand_max_roll_pitch_rate=0.20,
      # STOPPING lands the current swing, then places both diagonals at home.
      # Missing touchdown searches down at 5 cm/s, bounded to 8 cm.
      stand_foot_search_rate=0.05,
      stand_foot_search_depth=0.08,
      # STAND-only gains: latch the entry CoM/footholds and restore horizontal
      # drift without changing the force-dominant TROT stance controller.
      stand_mpc_xy_position_weight=5.0,
      stand_mpc_vxy_weight=3.0,
      stand_kp_cartesian=(80.0, 80.0, 30.0),
      stand_kd_cartesian=(8.0, 8.0, 8.0),
      # Internal STAND posture feedback only; this is not an RL joint-offset
      # action and the policy remains eight-dimensional foot placement only.
      stand_hipx_kp=12.0,
      stand_hipx_kd=1.0,
      # Keep fore-aft authority for mild terrain, but cap lateral residuals at
      # 3 cm. The previous 8 cm lateral range let the learned policy widen the
      # stance from 31.9 cm to as much as 47.9 cm and splay every HipX joint.
      foot_offset_scale_xy=(0.08, 0.03),
    )
  }

  # One term stores complete frames (not term-by-term history). Both groups
  # share the action term's once-per-policy-step cache, newest frame first.
  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    group.terms = {"state_history": ObservationTermCfg(func=mdp.mpc_state_history)}
    group.history_length = None
    group.enable_corruption = False

  twist = cast(UniformVelocityCommandCfg, cfg.commands["twist"])
  # These one-axis bounds each pass the zero-residual five-second gate on the
  # maximum 8 cm terrain. Stage one samples one command axis at a time so the
  # residual policy first learns a stable foothold baseline; mixed-command
  # fine-tuning can follow a stable checkpoint and a separate combined gate.
  twist.ranges.lin_vel_x = (-0.8, 0.8)
  twist.ranges.lin_vel_y = (-0.6, 0.6)
  twist.ranges.ang_vel_z = (-1.5, 1.5)
  twist.sample_single_axis_commands = True
  # Jointly train quiet four-leg standing and locomotion under one policy.
  # This gives the hybrid controller enough zero-command transitions without
  # letting stationary samples dominate velocity tracking.
  twist.rel_standing_envs = 0.25

  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=mdp.joint_torques_l2,
    weight=-2.5e-5,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"])},
  )
  cfg.rewards["foot_placement_offset_l2"] = RewardTermCfg(
    func=mdp.foot_placement_offset_l2,
    weight=-0.02,
  )
  cfg.rewards["foot_placement_lateral_offset_l2"] = RewardTermCfg(
    func=mdp.foot_placement_lateral_offset_l2,
    # The completed 8 cm policy saturated every dy action outward. Penalize
    # lateral residuals separately while preserving dx authority on terrain.
    weight=-0.05,
  )
  cfg.rewards["straight_foot_placement_symmetry"] = RewardTermCfg(
    func=mdp.straight_foot_placement_symmetry,
    weight=-0.1,
  )
  cfg.rewards["action_rate_l2"].weight = -0.02
  if "action_acc_l2" in cfg.rewards:
    cfg.rewards["action_acc_l2"].weight = -0.005

  # Keep the two failure signatures visible during retraining. Target a dy
  # action mean below 0.83 (2.5 cm) and a peak |HipX| below 0.20 rad.
  cfg.metrics["mean_abs_lateral_foot_placement_action"] = MetricsTermCfg(
    func=mdp.mean_abs_lateral_foot_placement_action,
  )
  cfg.metrics["max_abs_hip_x"] = MetricsTermCfg(
    func=mdp.max_abs_hip_x,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_HipX_joint",)),
    },
    reduce="max",
  )
  # Blind MPC retains contact sensors but consumes no ray/height sensors.
  scan_names = {"terrain_scan", "foot_height_scan", "base_height_scan"}
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in scan_names)
  for name, reward in list(cfg.rewards.items()):
    if any(
      value in scan_names for value in reward.params.values() if isinstance(value, str)
    ):
      del cfg.rewards[name]
  cfg.rewards["stopped_joint_pose_l2"] = RewardTermCfg(
    func=mdp.stopped_joint_pose_l2,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )

  # Remove the stair-specific performance gate. The generic terrain-distance
  # curriculum is retained for training and is the only route above flat level 0.
  terrain_levels = cfg.curriculum.get("terrain_levels")
  cfg.curriculum = (
    {"terrain_levels": terrain_levels} if terrain_levels is not None else {}
  )
  _remove_domain_randomization(cfg)

  if pure_mpc_validation:
    cfg.scene.num_envs = 3
    cfg.episode_length_s = 20.0
    cfg.curriculum = {}
    reset_base = cfg.events["reset_base"]
    reset_base.params["pose_range"] = {}
    reset_base.params["velocity_range"] = {}

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.curriculum = {}
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    # With >=3 play environments, proportional assignment guarantees at least
    # one flat, random and wave column. Levels span the full 0--8 cm bank.
    cfg.scene.terrain.max_init_terrain_level = 5

  # Assert the timing contract next to the final config to catch later edits.
  assert math.isclose(cfg.sim.mujoco.timestep * cfg.decimation, 0.02)
  return cfg


def syncai_g23_pure_mpc_validation_env_cfg() -> ManagerBasedRlEnvCfg:
  """Return the deterministic zero-residual configuration used by the gate tool."""
  return syncai_g23_rl_mpc_env_cfg(pure_mpc_validation=True)
