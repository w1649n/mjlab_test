"""Hospital navigation environment configuration.

A unicycle robot has to reach a goal in a hospital floor using a planar lidar,
optionally among pedestrians driven by a social force model. Ported from the
single-environment Stable-Baselines3 setup of "End-to-End Social Robot
Navigation with Reinforcement Learning".
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.navigation import assets, mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

# Scenarios the reference environment trains on.
TRAIN_SCENARIOS: tuple[str, ...] = (
  "scenario4",
  "scenario5",
  "scenario6",
  "scenario7",
  "scenario8",
  "scenario9",
  "scenario12",
  "scenarioTEST1",
  "scenarioTEST2",
  "scenarioTEST3",
)

NUM_LIDAR_RAYS = 108
LIDAR_MAX_DISTANCE = 20.0
CONTROL_DT = 0.25
EPISODE_LENGTH_S = 160.0
OBSERVATION_STACK = 2

# The reference rewards reach +200 / -70 on termination. Scaling everything down
# keeps the same trade-offs with value targets that PPO fits more easily.
_REWARD_SCALE = 0.1

_COMMAND = "goal"
_BASE_ACTION = "base_velocity"
_PEDESTRIAN_ACTION = "pedestrians"


def hospital_navigation_env_cfg(
  with_humans: bool = False, play: bool = False
) -> ManagerBasedRlEnvCfg:
  humans_name = "humans" if with_humans else None

  entities = {
    "hospital": assets.get_hospital_cfg(),
    "robot": assets.get_robot_cfg(),
  }
  if with_humans:
    entities["humans"] = assets.get_humans_cfg()

  lidar = RayCastSensorCfg(
    name="lidar",
    frame=ObjRef(type="site", name="lidar", entity="robot"),
    pattern=mdp.PlanarLidarPatternCfg(num_rays=NUM_LIDAR_RAYS),  # type: ignore[arg-type]
    max_distance=LIDAR_MAX_DISTANCE,
    debug_vis=play,
  )

  actor_terms = {
    "goal": ObservationTermCfg(
      func=mdp.goal_polar,
      params={"command_name": _COMMAND},
      scale=(0.1, 1.0 / math.pi),
    ),
    "lidar": ObservationTermCfg(
      func=mdp.lidar_scan,
      params={"sensor_name": "lidar"},
      scale=1.0 / LIDAR_MAX_DISTANCE,
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, history_length=OBSERVATION_STACK),
    "critic": ObservationGroupCfg({**actor_terms}, history_length=OBSERVATION_STACK),
  }

  actions: dict[str, ActionTermCfg] = {
    _BASE_ACTION: mdp.UnicycleActionCfg(
      entity_name="robot", max_lin_vel=1.0, max_ang_vel=1.0
    ),
  }
  if with_humans:
    actions[_PEDESTRIAN_ACTION] = mdp.PedestrianActionCfg(
      entity_name="humans",
      robot_action_name=_BASE_ACTION,
      # As in the reference setup, pedestrians only avoid the robot outside
      # of training.
      react_to_robot=play,
    )

  commands = {
    _COMMAND: mdp.NavigationCommandCfg(
      scenarios=TRAIN_SCENARIOS,
      pedestrian_action_name=_PEDESTRIAN_ACTION if with_humans else None,
      adaptive_sampling=not play,
      debug_vis=True,
    ),
  }

  rewards = {
    "progress": RewardTermCfg(
      func=mdp.goal_progress,
      weight=0.06 * _REWARD_SCALE,
      params={"command_name": _COMMAND},
    ),
    "bearing": RewardTermCfg(
      func=mdp.goal_bearing,
      weight=-0.05 * _REWARD_SCALE,
      params={"command_name": _COMMAND},
    ),
    "proximity": RewardTermCfg(
      func=mdp.obstacle_proximity,
      weight=-0.1 * _REWARD_SCALE,
      params={"humans_name": humans_name},
    ),
    "smoothness": RewardTermCfg(
      func=mdp.turning_smoothness,
      weight=-1.0 * _REWARD_SCALE,
      params={"action_name": _BASE_ACTION},
    ),
    "success": RewardTermCfg(
      func=mdp.goal_reached_bonus,
      weight=200.0 * _REWARD_SCALE,
      params={"command_name": _COMMAND, "humans_name": humans_name},
    ),
    "collision": RewardTermCfg(
      func=mdp.collision_penalty,
      weight=-70.0 * _REWARD_SCALE,
      params={"humans_name": humans_name},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "goal_reached": TerminationTermCfg(
      func=mdp.goal_reached, params={"command_name": _COMMAND}
    ),
    "collision": TerminationTermCfg(
      func=mdp.collision, params={"humans_name": humans_name}
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities=entities,
      sensors=(lidar,),
      num_envs=1,
      # Every environment uses the same hospital in its own world, so they all
      # share the world origin.
      env_spacing=0.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=18.0,
      elevation=-65.0,
      azimuth=90.0,
    ),
    # Robot and pedestrians are moved kinematically, so one physics step per
    # control step is enough and contacts are not needed.
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=CONTROL_DT, disableflags=("contact",)),
    ),
    decimation=1,
    episode_length_s=EPISODE_LENGTH_S,
    scale_rewards_by_dt=False,
  )


def navigation_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.7,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="navigation_hospital",
    save_interval=100,
    num_steps_per_env=64,
    max_iterations=3000,
  )
