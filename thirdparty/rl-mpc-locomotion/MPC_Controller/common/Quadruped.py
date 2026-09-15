from enum import Enum, auto
import math
from pathlib import Path
import numpy as np
import yaml
from MPC_Controller.utils import DTYPE

# Data structure containing parameters for quadruped robot
class RobotType(Enum):
    ALIENGO = auto()
    # MINI_CHEETAH = auto()
    A1 = auto()
    GO1 = auto()
    G23 = auto()

class Quadruped:

    def __init__(self, robotype:RobotType):

        if robotype is RobotType.ALIENGO:
            self._abadLinkLength = 0.083
            self._hipLinkLength = 0.25
            self._kneeLinkLength = 0.25
            self._kneeLinkY_offset = 0.0
            self._abadLocation = np.array([0.2399, 0.051, 0], dtype=DTYPE).reshape((3,1))
            self._bodyName = "trunk"
            self._bodyMass = 9.041 * 2
            self._bodyInertia = np.array([0.033260231, 0, 0, 
                                      0, 0.16117211, 0, 
                                      0, 0, 0.17460442])
            self._bodyHeight = 0.35
            self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            self._mpc_weights = np.array([1.0, 1.5, 0.0,
                                 0.0, 0.0, 50,
                                 0.0, 0.0, 0.1,
                                 1.0, 1.0, 0.1,
                                 0.0], dtype=DTYPE)
            # self._mpc_weights = [0.25, 0.25, 10, 2, 2, 50, 0, 0, 0.3, 0.2, 0.2, 0.1, 0]
            # self._mpc_weights = [1., 1., 0, 0, 0, 10, 0., 0., .1, .1, .1, .0, 0]

        elif robotype is RobotType.GO1:
            self._abadLinkLength = 0.08
            self._hipLinkLength = 0.213
            self._kneeLinkLength = 0.213
            self._kneeLinkY_offset = 0.0
            self._abadLocation = np.array([0.1881, 0.04675, 0], dtype=DTYPE).reshape((3,1))
            self._bodyName = "trunk"
            self._bodyMass = 5.204 * 2
            self._bodyInertia = np.array([0.0168128557, 0, 0, 
                                      0, 0.063009565, 0, 
                                      0, 0, 0.0716547275]) * 5
            self._bodyHeight = 0.26
            self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            self._mpc_weights = np.array([1.0, 1.5, 0.0,
                                 0.0, 0.0, 50,
                                 0.0, 0.0, 0.1,
                                 1.0, 1.0, 0.1,
                                 0.0], dtype=DTYPE) * 10

        elif robotype is RobotType.A1:
            self._abadLinkLength = 0.08505
            self._hipLinkLength = 0.2
            self._kneeLinkLength = 0.2
            self._kneeLinkY_offset = 0.0
            self._abadLocation = np.array([0.183, 0.047, 0], dtype=DTYPE).reshape((3,1))
            self._bodyName = "trunk"
            self._bodyMass = 8.5 * 3
            self._bodyInertia = np.array([0.017, 0, 0, 
                                      0, 0.057, 0, 
                                      0, 0, 0.064]) * 10
            self._bodyHeight = 0.26
            self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            # self._mpc_weights = [1., 1., 0, 0, 0, 20, 0., 0., .1, .1, .1, .0, 0]
            self._mpc_weights = np.array([0.25, 0.25, 10, 2, 2, 50, 0, 0, 0.3, 0.5, 0.5, 0.1, 0], dtype=DTYPE)

        elif robotype is RobotType.G23:
            self._abadLinkLength = 0.09735
            self._hipLinkLength = 0.20
            self._kneeLinkLength = 0.21012
            self._kneeLinkY_offset = 0.0
            self._abadLocation = np.array([0.1895, 0.062, 0], dtype=DTYPE).reshape((3,1))
            self._bodyName = "TORSO"
            self._bodyMass = 14.362548
            self._bodyInertia = np.array([0.040100, -0.000049, 0.001707,
                                      -0.000049, 0.121831, -0.000013,
                                      0.001707, -0.000013, 0.132349], dtype=DTYPE)
            self._bodyHeight = 0.34
            self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
            self._mpc_weights = np.array([100.0, 100.0, 70.0,
                                 0.0, 0.0, 25.0,
                                 0.2, 0.2, 1.0,
                                 0.3, 0.3, 0.5,
                                 0.0], dtype=DTYPE)

        # elif robotype is RobotType.MINI_CHEETAH:
        #     self._abadLinkLength = 0.062
        #     self._hipLinkLength = 0.209
        #     self._kneeLinkLength = 0.195
        #     self._kneeLinkY_offset = 0.004
        #     self._abadLocation = np.array([0.19, 0.049, 0], dtype=DTYPE).reshape((3,1))
        #     self._bodyName = "body"
        #     self._bodyMass = 3.3 * 3
        #     self._bodyInertia = np.array([0.011253, 0, 0, 
        #                               0, 0.036203, 0, 
        #                               0, 0, 0.042673]) * 10
        #     self._bodyHeight = 0.29
        #     self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
        #     # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
        #     self._mpc_weights = np.array([0.25, 0.25, 10, 2, 2, 50, 0, 0, 0.3, 0.2, 0.2, 0.1, 0], dtype=DTYPE)
        
        else:
            raise Exception("Invalid RobotType")
            
        self._robotType = robotype
        self._apply_task_yaml_overrides(robotype)

    def _apply_task_yaml_overrides(self, robotype: RobotType):
        task_names = {
            RobotType.ALIENGO: "Aliengo",
            RobotType.A1: "A1",
            RobotType.GO1: "Go1",
            RobotType.G23: "G23",
        }
        repo_root = Path(__file__).resolve().parents[2]
        task_yaml_path = repo_root / "RL_Environment" / "cfg" / "task" / f"{task_names[robotype]}.yaml"
        if not task_yaml_path.exists():
            return

        with task_yaml_path.open("r", encoding="utf-8") as config_file:
            task_cfg = yaml.safe_load(config_file) or {}

        body_height = task_cfg.get("env", {}).get("bodyHeight")
        if body_height is None:
            return

        body_height = float(body_height)
        if not math.isfinite(body_height) or body_height <= 0.0:
            raise ValueError(f"Invalid env.bodyHeight in {task_yaml_path}: {body_height}")

        self._bodyHeight = body_height

    def getHipLocation(self, leg:int):
        """
        Get location of the hip for the given leg in robot frame
        """
        assert leg >= 0 and leg < 4
        pHip = np.array([
            self._abadLocation[0] if (leg == 0 or leg == 1) else -self._abadLocation[0],
            self._abadLocation[1] if (leg == 0 or leg == 2) else -self._abadLocation[1],
            self._abadLocation[2]
            ], dtype=DTYPE).reshape((3,1))

        return pHip

    
