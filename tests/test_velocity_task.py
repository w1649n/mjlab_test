"""Tests specific to velocity tasks."""

from dataclasses import asdict
from typing import Any, cast

import pytest

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, G23_ACTION_SCALE, GO1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.g23.rl_cfg import syncai_g23_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_g23_stand_preserves_history6_observations_and_zero_command_play():
  original = load_env_cfg("Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6")
  task = "Mjlab-Velocity-Flat-SyncAI-G23-Proprio-History6-Stand"
  cfg = load_env_cfg(task)
  assert asdict(cfg.observations["actor"]) == asdict(original.observations["actor"])
  assert asdict(cfg.observations["critic"]) == asdict(original.observations["critic"])
  assert cfg.curriculum == {}
  assert "push_robot" not in cfg.events
  assert cfg.commands["twist"].rel_standing_envs == 0.7
  play_cfg = load_env_cfg(task, play=True)
  assert play_cfg.commands["twist"].rel_standing_envs == 1.0
  assert not play_cfg.commands["twist"].heading_command
  assert "randomize_terrain" not in play_cfg.events


def test_g23_stair_yaw_extends_history6_without_changing_observations():
  original = load_env_cfg("Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6")
  cfg = load_env_cfg("Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-StairYaw")
  assert asdict(cfg.observations["actor"]) == asdict(original.observations["actor"])
  assert asdict(cfg.observations["critic"]) == asdict(original.observations["critic"])
  stairs = ("pyramid_stairs", "pyramid_stairs_inv")

  reset_base = cfg.events["reset_base"]
  assert reset_base.func is mdp.reset_root_state_stair_aligned
  assert reset_base.params["terrain_names"] == stairs
  assert (
    reset_base.params["pose_range"]
    == (original.events["reset_base"].params["pose_range"])
  )

  heading = cfg.rewards["stair_heading_deviation"]
  assert heading.func is mdp.straight_heading_deviation_l2
  assert heading.weight < 0.0
  assert heading.params == {"command_name": "twist", "terrain_names": stairs}
  assert "stair_heading_deviation" not in original.rewards


def test_g23_stair_yaw_j60_limits_torque_and_penalizes_heating():
  base = load_env_cfg("Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-StairYaw")
  cfg = load_env_cfg("Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-StairYaw-J60")
  assert asdict(cfg.observations["actor"]) == asdict(base.observations["actor"])
  assert asdict(cfg.observations["critic"]) == asdict(base.observations["critic"])

  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  limits = {}
  for actuator in articulation.actuators:
    assert isinstance(actuator, DcMotorActuatorCfg)
    limits[actuator.target_names_expr] = (
      actuator.saturation_effort,
      actuator.effort_limit,
      actuator.velocity_limit,
      actuator.armature,
    )
  assert limits == {
    (r".*_Hip[XY]_joint",): (19.94, 19.94, 24.18, 0.004),
    (r".*_Knee_joint",): (30.5, 30.5, 15.49, 0.0089),
  }

  assert cfg.rewards["stair_heading_deviation"].weight == pytest.approx(-10.0)
  heat = cfg.rewards["motor_heat"]
  assert heat.func is mdp.motor_heat_l2
  assert heat.weight < 0.0
  assert heat.params["rated_torque"] == {
    r".*_Hip[XY]_joint": 6.09,
    r".*_Knee_joint": 9.58,
  }
  assert "motor_heat" not in base.rewards


def test_g23_stair_goals_extends_j60_for_the_stair_goals():
  base_id = "Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-StairYaw-J60"
  task_id = "Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-J60-StairGoals"
  base = load_env_cfg(base_id)
  cfg = load_env_cfg(task_id)
  # Same actor, and the critic keeps its layout with base-frame contact forces.
  assert asdict(cfg.observations["actor"]) == asdict(base.observations["actor"])
  critic = cfg.observations["critic"].terms
  assert list(critic) == list(base.observations["critic"].terms)
  assert critic["foot_contact_forces"].func is mdp.foot_contact_forces_b
  assert asdict(cfg.scene.entities["robot"]) == asdict(base.scene.entities["robot"])

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert twist.directional_heading_hold
  assert twist.heading_control_stiffness == pytest.approx(3.0)
  assert twist.rel_standing_envs == pytest.approx(0.2)
  fine_yaw = cfg.rewards["track_yaw_rate_fine"]
  assert fine_yaw.func is mdp.track_angular_velocity_yaw
  assert fine_yaw.params["std"] < cfg.rewards["track_angular_velocity"].params["std"]

  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  for name in ("pyramid_stairs", "pyramid_stairs_inv"):
    stairs = cast(Any, generator.sub_terrains[name])
    assert stairs.step_height_range == (0.05, 0.25)
  assert list(cfg.curriculum) == ["command_performance", "terrain_levels"]
  assert cfg.curriculum["terrain_levels"].func is mdp.terrain_levels_tracked
  assert cfg.rewards["foot_clearance"].func is mdp.feet_clearance_stair_adaptive
  assert cfg.rewards["foot_swing_height"].func is mdp.feet_swing_height_stair_adaptive

  deviation = cfg.rewards["standing_joint_deviation"]
  assert deviation.weight < 0.0
  assert deviation.params["foot_height_sensor_name"] == "foot_height_scan"
  fine_lin = cfg.rewards["track_linear_velocity_fine"]
  assert fine_lin.func is mdp.track_linear_velocity_xy
  assert fine_lin.params["std"] < cfg.rewards["track_linear_velocity"].params["std"]
  assert cfg.rewards["standing_joint_velocity"].params["settle_time_s"] > 0.0
  contacts = cfg.rewards["standing_missing_foot_contacts"]
  assert contacts.params["settle_time_s"] > 0.0
  heat = cfg.rewards["motor_heat_excess"]
  assert heat.func is mdp.motor_heat_l2
  assert heat.weight < 0.0 and heat.params["threshold"] == pytest.approx(1.0)
  assert "settle_time_s" not in base.rewards["standing_missing_foot_contacts"].params

  play = load_env_cfg(task_id, play=True)
  assert play.curriculum == {}
  rl_cfg, base_rl_cfg = load_rl_cfg(task_id), load_rl_cfg(base_id)
  assert isinstance(rl_cfg, RslRlOnPolicyRunnerCfg)
  assert isinstance(base_rl_cfg, RslRlOnPolicyRunnerCfg)
  symmetry_cfg = rl_cfg.algorithm.symmetry_cfg
  assert symmetry_cfg is not None and symmetry_cfg["use_data_augmentation"]
  assert rl_cfg.algorithm.entropy_coef == pytest.approx(0.003)
  assert base_rl_cfg.algorithm.symmetry_cfg is None


@pytest.fixture(scope="module")
def velocity_task_ids() -> list[str]:
  """Get all velocity task IDs."""
  return [t for t in list_tasks() if "Velocity" in t]


@pytest.fixture(scope="module")
def g1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all G1 velocity task IDs."""
  return [t for t in velocity_task_ids if "G1" in t]


@pytest.fixture(scope="module")
def go1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all Go1 velocity task IDs."""
  return [t for t in velocity_task_ids if "Go1" in t]


@pytest.fixture(scope="module")
def g23_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all G23 velocity task IDs."""
  return [t for t in velocity_task_ids if "G23" in t]


@pytest.fixture(scope="module")
def rough_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all rough terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Rough" in t]


@pytest.fixture(scope="module")
def flat_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all flat terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Flat" in t]


def test_velocity_tasks_have_twist_command(velocity_task_ids: list[str]) -> None:
  """All velocity tasks should have a velocity command."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "twist" in cfg.commands, f"Task {task_id} missing 'twist' command"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg), (
      f"Task {task_id} twist command is not UniformVelocityCommandCfg"
    )


def test_g1_velocity_has_required_sensors(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should have feet/ground and self collision sensors."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_go1_velocity_has_required_sensors(go1_velocity_task_ids: list[str]) -> None:
  """Go1 velocity tasks should have feet/ground and collision sensors."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    if "Rough" in task_id:
      for name in (
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
      ):
        assert name in sensor_names, f"Task {task_id} missing {name} sensor"


def test_g23_velocity_has_required_sensors(g23_velocity_task_ids: list[str]) -> None:
  """G23 velocity tasks should have feet/ground and collision sensors."""
  for task_id in g23_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    if "Rough" in task_id:
      for name in (
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
        "torso_ground_touch",
      ):
        assert name in sensor_names, f"Task {task_id} missing {name} sensor"


def test_flat_velocity_tasks_have_plane_terrain(
  flat_velocity_task_ids: list[str],
) -> None:
  """Flat velocity tasks should have terrain_type='plane' and no terrain_generator."""
  for task_id in flat_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "plane", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, expected 'plane'"
    )
    assert cfg.scene.terrain.terrain_generator is None, (
      f"Task {task_id} has terrain_generator, expected None for flat terrain"
    )


def test_rough_velocity_tasks_have_generator_terrain(
  rough_velocity_task_ids: list[str],
) -> None:
  """Rough velocity tasks should have generator terrain."""
  for task_id in rough_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "generator", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, "
      "expected 'generator'"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator, expected one for rough terrain"
    )


def test_rough_velocity_training_has_curriculum_enabled() -> None:
  """Rough velocity training tasks should have terrain curriculum enabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
    "Mjlab-Velocity-Rough-SyncAI-G23",
    "Mjlab-Velocity-Rough-SyncAI-G23-Proprio",
    "Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is True, (
      f"Task {task_id} curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected True"
    )


def test_rough_velocity_play_has_curriculum_disabled() -> None:
  """Rough velocity play tasks should have terrain curriculum disabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
    "Mjlab-Velocity-Rough-SyncAI-G23",
    "Mjlab-Velocity-Rough-SyncAI-G23-Proprio",
    "Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id, play=True)

    assert cfg.scene.terrain is not None, (
      f"Task {task_id} (play mode) has no terrain config"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} (play mode) has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is False, (
      f"Task {task_id} (play mode) curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected False"
    )


def test_g1_velocity_has_correct_action_scale(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should use G1_ACTION_SCALE."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_go1_velocity_has_correct_action_scale(
  go1_velocity_task_ids: list[str],
) -> None:
  """Go1 velocity tasks should use GO1_ACTION_SCALE."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == GO1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected GO1_ACTION_SCALE"
    )


def test_g23_velocity_has_correct_action_scale(
  g23_velocity_task_ids: list[str],
) -> None:
  """G23 velocity tasks should use G23_ACTION_SCALE."""
  for task_id in g23_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G23_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G23_ACTION_SCALE"
    )


def test_g23_proprio_actor_observation_order() -> None:
  """Every proprioceptive G23 task should use the compact 45-D frame layout."""
  original_cfg = load_env_cfg("Mjlab-Velocity-Rough-SyncAI-G23")

  original_actor_terms = list(original_cfg.observations["actor"].terms)
  assert "base_lin_vel" in original_actor_terms
  assert "height_scan" in original_actor_terms

  for task_id in (
    "Mjlab-Velocity-Rough-SyncAI-G23-Proprio",
    "Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6",
    "Mjlab-Velocity-Flat-SyncAI-G23-Proprio",
  ):
    proprio_cfg = load_env_cfg(task_id)
    assert list(proprio_cfg.observations["actor"].terms) == [
      "base_ang_vel",
      "projected_gravity",
      "command",
      "joint_pos",
      "joint_vel",
      "actions",
    ]


def test_g23_history6_is_opt_in_and_actor_only() -> None:
  history_cfg = load_env_cfg("Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6")
  actor = history_cfg.observations["actor"]
  critic = history_cfg.observations["critic"]
  assert actor.history_length == 6
  assert actor.flatten_history_dim is True
  assert critic.history_length is None

  legacy_cfg = load_env_cfg("Mjlab-Velocity-Rough-SyncAI-G23-Proprio")
  assert legacy_cfg.observations["actor"].history_length is None


def test_g23_flat_proprio_removes_rough_terrain_dependencies() -> None:
  """The flat G23 task should not retain rough-terrain-only configuration."""
  task_id = "Mjlab-Velocity-Flat-SyncAI-G23-Proprio"
  assert task_id in list_tasks()

  cfg = load_env_cfg(task_id)
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.terrain.terrain_generator is None

  sensor_names = {sensor.name for sensor in (cfg.scene.sensors or ())}
  assert sensor_names.isdisjoint(
    {
      "terrain_scan",
      "self_collision",
      "thigh_ground_touch",
      "shank_ground_touch",
      "torso_ground_touch",
    }
  )
  assert "height_scan" not in cfg.observations["critic"].terms
  assert "terrain_sensor_names" not in cfg.rewards["upright"].params
  assert "terrain_levels" not in cfg.curriculum
  assert "command_performance" not in cfg.curriculum
  assert "out_of_terrain_bounds" not in cfg.terminations
  assert "fell_over" in cfg.terminations
  assert "excessive_foot_air_time" in cfg.rewards
  assert "standing_missing_foot_contacts" in cfg.rewards


def test_g23_stair_training_update() -> None:
  """G23 stair training should be performance-gated and conservatively tuned."""
  cfg = load_env_cfg("Mjlab-Velocity-Rough-SyncAI-G23-Proprio")

  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  proportions = {
    name: generator.sub_terrains[name].proportion
    for name in (
      "flat",
      "pyramid_stairs",
      "pyramid_stairs_inv",
      "hf_pyramid_slope",
      "hf_pyramid_slope_inv",
      "random_rough",
      "wave_terrain",
    )
  }
  assert proportions == pytest.approx(
    {
      "flat": 0.10,
      "pyramid_stairs": 0.40,
      "pyramid_stairs_inv": 0.40,
      "hf_pyramid_slope": 0.025,
      "hf_pyramid_slope_inv": 0.025,
      "random_rough": 0.05,
      "wave_terrain": 0.00,
    }
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  assert twist_cmd.rel_forward_envs == pytest.approx(0.4)
  assert twist_cmd.rel_backward_envs == pytest.approx(0.1)
  assert twist_cmd.ranges.lin_vel_x == (-1.0, 1.0)

  assert list(cfg.curriculum) == ["command_performance", "terrain_levels"]
  command_curriculum = cfg.curriculum["command_performance"]
  assert command_curriculum.func is mdp.PerformanceGatedVelocityCurriculum
  stages = command_curriculum.params["stages"]
  assert stages[-1]["lin_vel_x"] == (-1.5, 1.5)
  required_stage_fields = {
    "min_terrain_level",
    "lin_vel_x",
    "lin_vel_y",
    "ang_vel_z",
    "rel_forward_envs",
    "rel_backward_envs",
  }
  for stage in stages:
    assert set(stage) == required_stage_fields
    assert "step" not in stage
    lin_vel_x = stage["lin_vel_x"]
    assert lin_vel_x[0] >= -1.5
    assert lin_vel_x[1] <= 1.5
  assert [stage["min_terrain_level"] for stage in stages] == [0, 3, 5, 7, 9]
  assert [stage["rel_backward_envs"] for stage in stages] == [
    0.1,
    0.2,
    0.2,
    0.3,
    0.3,
  ]
  assert command_curriculum.params["required_passes"] == 2
  assert command_curriculum.params["min_stage_steps"] == 500 * 24
  assert command_curriculum.params["contact_sensor_name"] == "feet_ground_contact"
  assert command_curriculum.params["max_foot_air_time"] == pytest.approx(0.8)

  runner_cfg = syncai_g23_ppo_runner_cfg()
  assert runner_cfg.clip_actions == pytest.approx(5.0)

  shank_reward = cfg.rewards["shank_collision"]
  assert shank_reward.weight == -0.25
  assert shank_reward.params["force_threshold"] == 10.0

  assert cfg.rewards["track_linear_velocity"].func is mdp.track_linear_velocity_xy
  assert cfg.rewards["track_angular_velocity"].func is mdp.track_angular_velocity_yaw
  assert cfg.rewards["body_ang_vel"].weight == pytest.approx(-0.02)
  assert cfg.rewards["foot_clearance"].weight == pytest.approx(-0.75)
  assert cfg.rewards["foot_swing_height"].weight == pytest.approx(-0.25)
  excessive_air = cfg.rewards["excessive_foot_air_time"]
  assert excessive_air.func is mdp.feet_excessive_air_time
  assert excessive_air.weight == pytest.approx(-4.0)
  assert excessive_air.params == {
    "sensor_name": "feet_ground_contact",
    "max_air_time": 0.55,
    "max_excess_time": 0.5,
  }
  standing_contacts = cfg.rewards["standing_missing_foot_contacts"]
  assert standing_contacts.func is mdp.feet_contact_count_standing
  assert standing_contacts.weight == pytest.approx(-0.5)
  assert standing_contacts.params == {
    "sensor_name": "feet_ground_contact",
    "command_name": "twist",
    "required_contacts": 4,
    "command_threshold": 0.05,
  }
  assert cfg.rewards["air_time"].weight == 0.0
  assert cfg.rewards["action_rate_l2"].weight == pytest.approx(-0.05)
  assert cfg.rewards["action_acc_l2"].weight == pytest.approx(-0.01)

  sensors = {sensor.name: sensor for sensor in (cfg.scene.sensors or ())}
  assert "base_height_scan" in sensors
  for name in ("thigh_ground_touch", "shank_ground_touch", "torso_ground_touch"):
    assert sensors[name].reduce == "maxforce"

  thigh_termination = cfg.terminations["illegal_contact"]
  assert thigh_termination.params["force_threshold"] == 60.0
  assert thigh_termination.params["history_count_threshold"] == 2
  shank_termination = cfg.terminations["shank_illegal_contact"]
  assert shank_termination.params["force_threshold"] == 180.0
  assert shank_termination.params["history_count_threshold"] == 4
  torso_termination = cfg.terminations["torso_illegal_contact"]
  assert torso_termination.params["force_threshold"] == 80.0
  assert torso_termination.params["history_count_threshold"] == 3
  foot_air_termination = cfg.terminations["prolonged_foot_air_time"]
  assert foot_air_termination.func is mdp.prolonged_foot_air_time
  assert foot_air_termination.params == {
    "sensor_name": "feet_ground_contact",
    "max_air_time": 2.0,
  }

  push = cfg.events["push_robot"]
  assert push.mode == "step"
  assert push.func is mdp.apply_body_impulse
  assert push.params["duration_s"] == (0.10, 0.15)
  assert push.params["cooldown_s"] == (8.0, 12.0)


def test_velocity_tasks_do_not_share_sub_terrain_configs():
  """G23 stair tuning must not leak into other robots or other G23 tasks."""
  go1 = load_env_cfg("Mjlab-Velocity-Rough-Unitree-Go1")
  goals = load_env_cfg(
    "Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-J60-StairGoals"
  )
  j60 = load_env_cfg("Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-StairYaw-J60")
  ranges = []
  for cfg in (go1, goals, j60):
    assert cfg.scene.terrain is not None
    generator = cfg.scene.terrain.terrain_generator
    assert generator is not None
    ranges.append(cast(Any, generator.sub_terrains["pyramid_stairs"]).step_height_range)
  assert ranges == [(0.0, 0.1), (0.05, 0.25), (0.02, 0.2)]
