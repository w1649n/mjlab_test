"""SyncAI G23 constants and mjlab entity configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G23_XML: Path = Path(__file__).parent / "xmls" / "g23.xml"
assert G23_XML.exists()

G23_FOOT_SITE_NAMES = ("FL", "FR", "HL", "HR")
_FOOT_BODY_NAMES = tuple(f"{leg}_FOOT" for leg in G23_FOOT_SITE_NAMES)


def get_spec() -> mujoco.MjSpec:
  """Load a fresh copy of the G23 MJCF specification."""
  spec = mujoco.MjSpec.from_file(str(G23_XML))
  for site_name, body_name in zip(G23_FOOT_SITE_NAMES, _FOOT_BODY_NAMES, strict=True):
    spec.body(body_name).add_site(
      name=site_name,
      pos=(0.0, 0.0, 0.0),
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(0.01,),
      group=5,
    )
  return spec


##
# Actuator configuration.
#
# Values follow the existing SyncAI G23 locomotion configuration. The source model
# specifies 26.2 rad/s and 24 Nm for hip motors, and 17.3 rad/s and 36 Nm for knee
# motors. BuiltinPositionActuatorCfg enforces the effort limits; the velocity values
# are retained in this comment because this actuator type has no velocity-limit field.
##

G23_HIP_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_Hip[XY]_joint",),
  stiffness=30.0,
  damping=1.0,
  effort_limit=24.0,
  # Keep the reflected inertia declared by the active MJCF.
  armature=None,
  delay_min_lag=0,
  delay_max_lag=1,
)

G23_KNEE_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_Knee_joint",),
  stiffness=30.0,
  damping=1.0,
  effort_limit=36.0,
  # Keep the reflected inertia declared by the active MJCF.
  armature=None,
  delay_min_lag=0,
  delay_max_lag=1,
)


##
# Initial state.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.375),
  joint_pos={
    r".*_HipX_joint": 0.0,
    r".*_HipY_joint": -0.65,
    r".*_Knee_joint": 1.3,
  },
  joint_vel={r".*": 0.0},
)


##
# Collision configuration.
##

_foot_regex = r"^(FL|FR|HL|HR)_foot_collision$"
_collision_regex = r".*_collision\d*"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_collision_regex,),
  contype=1,
  conaffinity=1,
  solref=(0.01, 1),
  condim={
    _foot_regex: 6,
    _collision_regex: 1,
  },
  priority={
    _foot_regex: 1,
    _collision_regex: 0,
  },
  friction={
    _foot_regex: (1.0, 5e-3, 5e-4),
  },
)


##
# Final entity configuration.
##

G23_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(G23_HIP_ACTUATOR, G23_KNEE_ACTUATOR),
  soft_joint_pos_limit_factor=0.99,
)


def get_g23_robot_cfg() -> EntityCfg:
  """Return a fresh G23 robot configuration."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G23_ARTICULATION,
    sort_actuators=True,
  )


# Default-pose-relative action scale used by the existing locomotion setup.
G23_ACTION_SCALE: dict[str, float] = {
  r".*_HipX_joint": 0.125,
  r".*_HipY_joint": 0.25,
  r".*_Knee_joint": 0.25,
}

# The getup task uses targets relative to the *current* joint position, so its
# scale acts like a bounded torque request: kp * scale.  These values give every
# motor its full configured authority at a clipped action of +/-1.
G23_GETUP_ACTION_SCALE: dict[str, float] = {
  r".*_Hip[XY]_joint": 24.0 / 30.0,
  r".*_Knee_joint": 36.0 / 30.0,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity import Entity

  robot = Entity(get_g23_robot_cfg())
  viewer.launch(robot.compile())
