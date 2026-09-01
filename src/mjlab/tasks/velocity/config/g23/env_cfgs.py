"""SyncAI G23 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  G23_ACTION_SCALE,
  G23_FOOT_SITE_NAMES,
  get_g23_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
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
#first parameter: heightfield horizontal_scale = 0.1
_G23_HFIELD_HORIZONTAL_SCALE = 0.2
#first parameter: wave_terrain amplitude_range = (0.0, 0.2)
_G23_WAVE_AMPLITUDE_RANGE = (0.0, 0.08)
#first parameter: pyramid stair step_height_range = (0.0, 0.1)
_G23_STAIR_STEP_HEIGHT_RANGE = (0.02, 0.20)  # second parameter
#first parameter: pyramid stair step_width = 0.3
_G23_STAIR_STEP_WIDTH = 0.30  # second parameter
#first parameter: foot clearance and swing target_height = 0.1
_G23_FOOT_CLEARANCE_TARGET_HEIGHT = 0.16  # second parameter
#first parameter: cfg.scene.terrain.max_init_terrain_level = 5
_G23_MAX_INIT_TERRAIN_LEVEL = 3  # second parameter
#second parameter: maximum command lin_vel_x reached (-2.0, 3.0)
_G23_MAX_FORWARD_SPEED = 1.5  #third parameter
#second parameter: shank_collision weight = -0.1
_G23_SHANK_COLLISION_WEIGHT = -0.25  #third parameter
#second parameter: shank_collision used default force_threshold = 10.0
_G23_SHANK_PENALTY_FORCE_THRESHOLD = 10.0  #third parameter
#second parameter: shank_illegal_contact reset on any force > 10.0
_G23_SHANK_TERMINATION_FORCE_THRESHOLD = 120.0  #third parameter
#second parameter: shank_illegal_contact reset on any history hit
_G23_SHANK_TERMINATION_HISTORY_COUNT = 3  #third parameter
_RL_TRAIN_POLICY_TERM_ORDER = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "joint_pos",
  "joint_vel",
  "actions",
)


def _set_rl_train_policy_observation_order(cfg: ManagerBasedRlEnvCfg) -> None:
  """Match RL-train's proprioceptive actor observation order."""
  actor_terms = cfg.observations["actor"].terms
  cfg.observations["actor"].terms = {
    name: actor_terms[name] for name in _RL_TRAIN_POLICY_TERM_ORDER
  }


def _tune_g23_rough_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
  """Tune G23 rough terrain toward the 20 cm stair-climbing target."""
  terrain = cfg.scene.terrain
  if terrain is None or terrain.terrain_generator is None:
    return

  generator = terrain.terrain_generator
  generator.curriculum = True
  terrain.max_init_terrain_level = _G23_MAX_INIT_TERRAIN_LEVEL  # second parameter

  for sub_terrain in generator.sub_terrains.values():
    if hasattr(sub_terrain, "horizontal_scale"):
      sub_terrain.horizontal_scale = max(
        sub_terrain.horizontal_scale, _G23_HFIELD_HORIZONTAL_SCALE
      )

  #first parameter: inherited rough terrain proportions are
  #first parameter: flat=0.2, stairs=0.2/0.2, slopes=0.1/0.1, rough=0.1, wave=0.1
  #second parameter: flat=0.10, stairs=0.30/0.30, slopes=0.10/0.10, rough=0.10, wave=0.00
  terrain_proportions = {
    "flat": 0.10,  #third parameter
    "pyramid_stairs": 0.40,  #third parameter
    "pyramid_stairs_inv": 0.40,  #third parameter
    "hf_pyramid_slope": 0.025,  #third parameter
    "hf_pyramid_slope_inv": 0.025,  #third parameter
    "random_rough": 0.05,  #third parameter
    "wave_terrain": 0.00,  #third parameter
  }
  for terrain_name, proportion in terrain_proportions.items():
    sub_terrain = generator.sub_terrains.get(terrain_name)
    if sub_terrain is not None:
      sub_terrain.proportion = proportion

  for stair_name in ("pyramid_stairs", "pyramid_stairs_inv"):
    stair = generator.sub_terrains.get(stair_name)
    if stair is not None:
      stair.step_height_range = _G23_STAIR_STEP_HEIGHT_RANGE  # second parameter
      stair.step_width = _G23_STAIR_STEP_WIDTH  # second parameter

  wave_terrain = generator.sub_terrains.get("wave_terrain")
  if wave_terrain is not None and hasattr(wave_terrain, "amplitude_range"):
    wave_terrain.amplitude_range = _G23_WAVE_AMPLITUDE_RANGE


def _tune_g23_locomotion_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Tune G23 rough locomotion for better foot clearance before harder commands."""
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)

  #first parameter: twist_cmd.rel_forward_envs = 0.2
  # twist_cmd.rel_forward_envs = 0.6  # second parameter
  twist_cmd.rel_forward_envs = 0.4  # fourth parameter
  #first parameter: twist_cmd.rel_heading_envs = 0.3
  twist_cmd.rel_heading_envs = 0.1  # second parameter
  #second parameter: twist_cmd.ranges.lin_vel_x inherited base range before curriculum
  twist_cmd.ranges.lin_vel_x = (
    -_G23_MAX_FORWARD_SPEED,
    _G23_MAX_FORWARD_SPEED,
  )  #third parameter
  #first parameter: twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)  # second parameter

  command_curriculum = cfg.curriculum["command_vel"]
  #first parameter: velocity_stages = [
  #first parameter:   {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
  #first parameter:   {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
  #first parameter:   {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
  #first parameter: ]
  #second parameter: final velocity stage used lin_vel_x = (-2.0, 3.0)
  command_curriculum.params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-1.0, 1.0),  #third parameter
      "lin_vel_y": (-0.2, 0.2),  # second parameter
      "ang_vel_z": (-0.5, 0.5),
    },
    {
      "step": 8000 * 24,
      "lin_vel_x": (-1.25, 1.25),  #third parameter
      "lin_vel_y": (-0.4, 0.4),  # second parameter
      "ang_vel_z": (-0.7, 0.7),
    },
    {
      "step": 14000 * 24,
      "lin_vel_x": (
        -_G23_MAX_FORWARD_SPEED,
        _G23_MAX_FORWARD_SPEED,
      ),  #third parameter
      "lin_vel_y": (-0.6, 0.6),  # second parameter
    },
  ]


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
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(mode="geom", pattern=_SHANK_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  torso_ground_cfg = ContactSensorCfg(
    name="torso_ground_touch",
    primary=ContactMatch(mode="geom", pattern=_TORSO_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
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
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (_BASE_BODY_NAME,)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = G23_FOOT_SITE_NAMES

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0
  #first parameter: cfg.rewards["foot_clearance"].params["target_height"] = 0.1
  cfg.rewards["foot_clearance"].params["target_height"] = (
    _G23_FOOT_CLEARANCE_TARGET_HEIGHT
  )  # second parameter
  #first parameter: cfg.rewards["foot_swing_height"].params["target_height"] = 0.1
  cfg.rewards["foot_swing_height"].params["target_height"] = (
    _G23_FOOT_CLEARANCE_TARGET_HEIGHT
  )  # second parameter
  #first parameter: cfg.rewards["foot_clearance"].weight = -2.0
  cfg.rewards["foot_clearance"].weight = -3.0
  #first parameter: cfg.rewards["foot_swing_height"].weight = -0.25
  cfg.rewards["foot_swing_height"].weight = -0.75
  #first parameter: cfg.rewards["soft_landing"].weight = -1e-5
  cfg.rewards["soft_landing"].weight = -5e-5

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=_G23_SHANK_COLLISION_WEIGHT,  #third parameter
    params={
      "sensor_name": shank_ground_cfg.name,
      "force_threshold": _G23_SHANK_PENALTY_FORCE_THRESHOLD,  #third parameter
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
    params={"sensor_name": thigh_ground_cfg.name},
  )
  #first parameter: no shank ground-contact termination
  #second parameter: shank ground-contact termination used default force_threshold=10.0
  cfg.terminations["shank_illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": shank_ground_cfg.name,
      "force_threshold": _G23_SHANK_TERMINATION_FORCE_THRESHOLD,  #third parameter
      "history_count_threshold": _G23_SHANK_TERMINATION_HISTORY_COUNT,  #third parameter
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


def syncai_g23_rough_proprio_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create SyncAI G23 rough terrain config with RL-train policy observations."""
  cfg = syncai_g23_rough_env_cfg(play=play)
  _set_rl_train_policy_observation_order(cfg)
  return cfg
