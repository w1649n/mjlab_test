"""SyncAI G23 velocity environment configurations."""

import math
from typing import Any, cast

from mjlab.asset_zoo.robots import (
  G23_ACTION_SCALE,
  G23_FOOT_SITE_NAMES,
  get_g23_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

_BASE_BODY_NAME = "TORSO"
_LEG_NAMES = ("FL", "FR", "HL", "HR")
_FOOT_GEOM_NAMES = tuple(f"{leg}_foot_collision" for leg in _LEG_NAMES)
_THIGH_GEOM_NAMES = tuple(f"{leg}_thigh_collision" for leg in _LEG_NAMES)
_SHANK_GEOM_NAMES = tuple(
  f"{leg}_shank_collision{suffix}" for leg in _LEG_NAMES for suffix in ("", "1")
)
_TORSO_GEOM_NAMES = ("torso_collision", "torso_collision1")

# 2026-09-02 stair-training update: keep all G23 stair/recovery tuning in one
# named block so future fine-tuning does not depend on checkpoint iteration.
_G23_HFIELD_HORIZONTAL_SCALE = 0.2
_G23_WAVE_AMPLITUDE_RANGE = (0.0, 0.08)
_G23_STAIR_STEP_HEIGHT_RANGE = (0.02, 0.20)
_G23_STAIR_STEP_WIDTH = 0.30
_G23_FOOT_CLEARANCE_TARGET_HEIGHT = 0.16
_G23_MAX_INIT_TERRAIN_LEVEL = 3
_G23_MAX_FORWARD_SPEED = 1.5
_G23_SHANK_COLLISION_WEIGHT = -0.25
_G23_SHANK_PENALTY_FORCE_THRESHOLD = 10.0
_G23_THIGH_TERMINATION_FORCE_THRESHOLD = 60.0
_G23_THIGH_TERMINATION_HISTORY_COUNT = 2
_G23_SHANK_TERMINATION_FORCE_THRESHOLD = 180.0
_G23_SHANK_TERMINATION_HISTORY_COUNT = 4
_G23_TORSO_TERMINATION_FORCE_THRESHOLD = 80.0
_G23_TORSO_TERMINATION_HISTORY_COUNT = 3
_G23_BASE_HEIGHT_TARGET = 0.35
# 2026-09-02 anti-tripod update: bound the cost while making a persistently
# airborne foot more expensive than avoiding a normal landing.
_G23_MAX_FOOT_AIR_TIME = 0.55
_G23_MAX_FOOT_AIR_EXCESS_TIME = 0.50
_G23_GATE_MAX_FOOT_AIR_TIME = 0.80
_G23_TERMINATION_MAX_FOOT_AIR_TIME = 2.0
_G23_EXCESSIVE_FOOT_AIR_TIME_WEIGHT = -4.0
_G23_STANDING_MISSING_CONTACT_WEIGHT = -0.5

_G23_PROPRIO_TERM_ORDER = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "joint_pos",
  "joint_vel",
  "actions",
)


def _set_g23_proprio_observation_order(cfg: ManagerBasedRlEnvCfg) -> None:
  """Use the compact 45-D G23 proprioceptive actor observation."""
  actor_terms = cfg.observations["actor"].terms
  cfg.observations["actor"].terms = {
    name: actor_terms[name] for name in _G23_PROPRIO_TERM_ORDER
  }


def _tune_g23_rough_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
  """Tune G23 rough terrain toward the 20 cm stair-climbing target."""
  terrain = cfg.scene.terrain
  if terrain is None or terrain.terrain_generator is None:
    return

  generator = terrain.terrain_generator
  generator.curriculum = True
  terrain.max_init_terrain_level = _G23_MAX_INIT_TERRAIN_LEVEL

  for sub_terrain in generator.sub_terrains.values():
    if hasattr(sub_terrain, "horizontal_scale"):
      heightfield = cast(Any, sub_terrain)
      heightfield.horizontal_scale = max(
        heightfield.horizontal_scale, _G23_HFIELD_HORIZONTAL_SCALE
      )

  # 2026-09-02 stair-training update: focus 80% of generated patches on both
  # stair directions while retaining a small non-stair robustness set.
  terrain_proportions = {
    "flat": 0.10,
    "pyramid_stairs": 0.40,
    "pyramid_stairs_inv": 0.40,
    "hf_pyramid_slope": 0.025,
    "hf_pyramid_slope_inv": 0.025,
    "random_rough": 0.05,
    "wave_terrain": 0.00,
  }
  for terrain_name, proportion in terrain_proportions.items():
    sub_terrain = generator.sub_terrains.get(terrain_name)
    if sub_terrain is not None:
      sub_terrain.proportion = proportion

  for stair_name in ("pyramid_stairs", "pyramid_stairs_inv"):
    stair = cast(Any, generator.sub_terrains.get(stair_name))
    if stair is not None:
      stair.step_height_range = _G23_STAIR_STEP_HEIGHT_RANGE
      stair.step_width = _G23_STAIR_STEP_WIDTH

  wave_terrain = generator.sub_terrains.get("wave_terrain")
  if wave_terrain is not None and hasattr(wave_terrain, "amplitude_range"):
    cast(Any, wave_terrain).amplitude_range = _G23_WAVE_AMPLITUDE_RANGE


def _tune_g23_locomotion_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Gate harder G23 commands on demonstrated stair performance."""
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)

  # 2026-09-02 stair-training update: the old absolute-iteration schedule could
  # increase speed/backward exposure while stair performance was regressing.
  # Start every run from a complete easy snapshot; a stateful gate below is the
  # only path to harder commands and is restored when full training resumes.
  twist_cmd.rel_forward_envs = 0.4
  twist_cmd.rel_backward_envs = 0.1
  twist_cmd.rel_heading_envs = 0.1
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  performance_stages = [
    {
      "min_terrain_level": 0,
      "lin_vel_x": (-1.0, 1.0),
      "lin_vel_y": (-0.2, 0.2),
      "ang_vel_z": (-0.5, 0.5),
      "rel_forward_envs": 0.4,
      "rel_backward_envs": 0.1,
    },
    {
      "min_terrain_level": 3,
      "lin_vel_x": (-1.0, 1.0),
      "lin_vel_y": (-0.2, 0.2),
      "ang_vel_z": (-0.5, 0.5),
      "rel_forward_envs": 0.4,
      "rel_backward_envs": 0.2,
    },
    {
      "min_terrain_level": 5,
      "lin_vel_x": (-1.25, 1.25),
      "lin_vel_y": (-0.4, 0.4),
      "ang_vel_z": (-0.7, 0.7),
      "rel_forward_envs": 0.4,
      "rel_backward_envs": 0.2,
    },
    {
      "min_terrain_level": 7,
      "lin_vel_x": (-1.25, 1.25),
      "lin_vel_y": (-0.4, 0.4),
      "ang_vel_z": (-0.7, 0.7),
      "rel_forward_envs": 0.4,
      "rel_backward_envs": 0.3,
    },
    {
      "min_terrain_level": 9,
      "lin_vel_x": (-_G23_MAX_FORWARD_SPEED, _G23_MAX_FORWARD_SPEED),
      "lin_vel_y": (-0.6, 0.6),
      "ang_vel_z": (-0.7, 0.7),
      "rel_forward_envs": 0.4,
      "rel_backward_envs": 0.3,
    },
  ]

  # Dict order is intentional: collect terminal performance against the old
  # terrain level/origin before terrain_levels mutates either one.
  terrain_curriculum = cfg.curriculum["terrain_levels"]
  cfg.curriculum = {
    "command_performance": CurriculumTermCfg(
      func=mdp.PerformanceGatedVelocityCurriculum,
      params={
        "command_name": "twist",
        "stages": performance_stages,
        "terrain_names": ("pyramid_stairs", "pyramid_stairs_inv"),
        "min_episodes_per_terrain": 256,
        "min_directional_samples_per_terrain": 20_000,
        "min_safe_success_rate": 0.60,
        "min_directional_tracking_score": 0.60,
        "warmup_steps": 1_000,
        "min_stage_steps": 500 * 24,
        "required_passes": 2,
        "success_termination_name": "time_out",
        # 2026-09-02 anti-tripod update: a policy cannot unlock harder commands
        # while finishing episodes with one foot persistently held in the air.
        "contact_sensor_name": "feet_ground_contact",
        "max_foot_air_time": _G23_GATE_MAX_FOOT_AIR_TIME,
      },
    ),
    "terrain_levels": terrain_curriculum,
  }


def syncai_g23_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create SyncAI G23 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 16
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 256

  cfg.scene.entities = {"robot": get_g23_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = _BASE_BODY_NAME

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in G23_FOOT_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=4)

  # 2026-09-02 stair-training update: measure torso clearance against the local
  # terrain instead of using world Z, which changes between stair levels.
  base_height_scan_cfg = TerrainHeightSensorCfg(
    name="base_height_scan",
    frame=ObjRef(type="body", name=_BASE_BODY_NAME, entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.3, 0.4), resolution=0.05),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    reduction="mean",
    debug_vis=False,
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=_FOOT_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=_BASE_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=_BASE_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(mode="geom", pattern=_THIGH_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(mode="geom", pattern=_SHANK_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  torso_ground_cfg = ContactSensorCfg(
    name="torso_ground_touch",
    primary=ContactMatch(mode="geom", pattern=_TORSO_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    base_height_scan_cfg,
    feet_ground_cfg,
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    torso_ground_cfg,
  )

  _tune_g23_rough_terrain(cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G23_ACTION_SCALE

  cfg.viewer.body_name = _BASE_BODY_NAME
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.45
  _tune_g23_locomotion_curriculum(cfg)

  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOM_NAMES),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOM_NAMES),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOM_NAMES),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = (_BASE_BODY_NAME,)

  # 2026-09-02 stair-training update: replace the frequent mass-independent
  # six-axis velocity kick with a sparse, short physical impulse on the torso.
  cfg.events["push_robot"] = EventTermCfg(
    mode="step",
    func=envs_mdp.apply_body_impulse,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_BODY_NAME,)),
      "force_range": (-20.0, 20.0),
      "torque_range": (0.0, 0.0),
      "duration_s": (0.10, 0.15),
      "cooldown_s": (8.0, 12.0),
    },
  )

  cfg.rewards["pose"].params["std_standing"] = {
    r".*_HipX_joint": 0.05,
    r".*_HipY_joint": 0.05,
    r".*_Knee_joint": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_HipX_joint": 0.2,
    r".*_HipY_joint": 0.3,
    r".*_Knee_joint": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_HipX_joint": 0.25,
    r".*_HipY_joint": 0.4,
    r".*_Knee_joint": 0.8,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = (_BASE_BODY_NAME,)
  # 2026-09-02 stair-training update: stay upright relative to gravity. Aligning
  # the torso to the stair-scan normal encouraged leaning into the staircase.
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (_BASE_BODY_NAME,)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = G23_FOOT_SITE_NAMES

  # 2026-09-02 stair-training update: tracking rewards now cover only their
  # commanded axes; vertical/balance/smoothness costs remain explicit and tunable.
  cfg.rewards["track_linear_velocity"].func = mdp.track_linear_velocity_xy
  cfg.rewards["track_angular_velocity"].func = mdp.track_angular_velocity_yaw
  cfg.rewards["body_ang_vel"].weight = -0.02
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0
  cfg.rewards["foot_clearance"].params["target_height"] = (
    _G23_FOOT_CLEARANCE_TARGET_HEIGHT
  )
  cfg.rewards["foot_swing_height"].params["target_height"] = (
    _G23_FOOT_CLEARANCE_TARGET_HEIGHT
  )
  # 2026-09-02 anti-tripod update: these landing/motion-gated costs were too
  # strong and could be avoided by lifting one foot once and never landing it.
  cfg.rewards["foot_clearance"].weight = -0.75
  cfg.rewards["foot_swing_height"].weight = -0.25
  cfg.rewards["soft_landing"].weight = -5e-5
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["excessive_foot_air_time"] = RewardTermCfg(
    func=mdp.feet_excessive_air_time,
    weight=_G23_EXCESSIVE_FOOT_AIR_TIME_WEIGHT,
    params={
      "sensor_name": feet_ground_cfg.name,
      "max_air_time": _G23_MAX_FOOT_AIR_TIME,
      "max_excess_time": _G23_MAX_FOOT_AIR_EXCESS_TIME,
    },
  )
  cfg.rewards["standing_missing_foot_contacts"] = RewardTermCfg(
    func=mdp.feet_contact_count_standing,
    weight=_G23_STANDING_MISSING_CONTACT_WEIGHT,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "twist",
      "required_contacts": len(G23_FOOT_SITE_NAMES),
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["vertical_velocity_l2"] = RewardTermCfg(
    func=mdp.vertical_velocity_l2,
    weight=-0.5,
  )
  cfg.rewards["base_height_l2"] = RewardTermCfg(
    func=mdp.base_height_l2,
    weight=-1.0,
    params={
      "height_sensor_name": base_height_scan_cfg.name,
      "target_height": _G23_BASE_HEIGHT_TARGET,
    },
  )
  cfg.rewards["electrical_power"] = RewardTermCfg(
    func=mdp.electrical_power_cost,
    weight=-2e-5,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-2.5e-7,
  )
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_acc_l2,
    weight=-0.01,
  )

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=_G23_SHANK_COLLISION_WEIGHT,
    params={
      "sensor_name": shank_ground_cfg.name,
      "force_threshold": _G23_SHANK_PENALTY_FORCE_THRESHOLD,
    },
  )
  cfg.rewards["torso_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": torso_ground_cfg.name},
  )

  cfg.terminations.pop("fell_over", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": thigh_ground_cfg.name,
      "force_threshold": _G23_THIGH_TERMINATION_FORCE_THRESHOLD,
      "history_count_threshold": _G23_THIGH_TERMINATION_HISTORY_COUNT,
    },
  )
  # 2026-09-02 stair-training update: recoverable leg brushes are penalized, but
  # only sustained hard contacts terminate; actual torso support terminates sooner.
  cfg.terminations["shank_illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": shank_ground_cfg.name,
      "force_threshold": _G23_SHANK_TERMINATION_FORCE_THRESHOLD,
      "history_count_threshold": _G23_SHANK_TERMINATION_HISTORY_COUNT,
    },
  )
  cfg.terminations["torso_illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": torso_ground_cfg.name,
      "force_threshold": _G23_TORSO_TERMINATION_FORCE_THRESHOLD,
      "history_count_threshold": _G23_TORSO_TERMINATION_HISTORY_COUNT,
    },
  )
  cfg.terminations["prolonged_foot_air_time"] = TerminationTermCfg(
    func=mdp.prolonged_foot_air_time,
    params={
      "sensor_name": feet_ground_cfg.name,
      "max_air_time": _G23_TERMINATION_MAX_FOOT_AIR_TIME,
    },
  )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def syncai_g23_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create SyncAI G23 flat terrain velocity configuration."""
  cfg = syncai_g23_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "torso_ground_touch",
  }
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name not in remove_sensors
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  for reward_name in ("self_collisions", "shank_collision", "torso_collision"):
    cfg.rewards.pop(reward_name, None)

  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("shank_illegal_contact", None)
  cfg.terminations.pop("torso_illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # The flat compatibility/fine-tuning task keeps the stage-0 command snapshot,
  # but has no stair evidence with which to advance the performance gate.
  cfg.curriculum.pop("command_performance", None)
  cfg.curriculum.pop("terrain_levels", None)

  return cfg


def syncai_g23_rough_proprio_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create SyncAI G23 rough terrain config with compact proprioception."""
  cfg = syncai_g23_rough_env_cfg(play=play)
  _set_g23_proprio_observation_order(cfg)
  return cfg


def syncai_g23_rough_proprio_history6_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G23 stair task with six proprioceptive frames for the actor."""
  cfg = syncai_g23_rough_proprio_env_cfg(play=play)
  # 2026-09-02 stair-training update: history is a separate task because a
  # 270-D actor cannot load the existing 45-D proprioceptive checkpoints.
  actor = cfg.observations["actor"]
  actor.history_length = 6
  actor.flatten_history_dim = True
  return cfg


def syncai_g23_flat_stand_history6_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Fine-tune stopping in the default stance, retaining checkpoint observations."""
  cfg = syncai_g23_rough_proprio_history6_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  # Keep the rough-task sensors and critic terms for checkpoint compatibility.
  cfg.curriculum = {}
  cfg.events.pop("randomize_terrain", None)
  cfg.events.pop("push_robot", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.rel_standing_envs = 1.0 if play else 0.7
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_heading_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.rel_backward_envs = 0.0
  twist.init_velocity_prob = 0.0
  twist.ranges.lin_vel_x = (-0.4, 0.4)
  twist.ranges.lin_vel_y = (-0.2, 0.2)
  twist.ranges.ang_vel_z = (-0.3, 0.3)
  twist.resampling_time_range = (4.0, 8.0)

  cfg.rewards["standing_joint_deviation"] = RewardTermCfg(
    func=mdp.standing_joint_deviation_l2,
    weight=-3.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["standing_joint_velocity"] = RewardTermCfg(
    func=mdp.standing_joint_velocity_l2,
    weight=-0.05,
    params={"command_name": "twist"},
  )
  cfg.rewards["standing_missing_foot_contacts"].weight = -2.0
  # Include zero commands in the existing foot-slip cost.
  cfg.rewards["foot_slip"].params["command_threshold"] = -1.0
  cfg.rewards["body_ang_vel"].weight = -0.1
  return cfg


def syncai_g23_flat_proprio_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create SyncAI G23 flat terrain config with compact proprioception."""
  cfg = syncai_g23_flat_env_cfg(play=play)
  _set_g23_proprio_observation_order(cfg)
  return cfg
