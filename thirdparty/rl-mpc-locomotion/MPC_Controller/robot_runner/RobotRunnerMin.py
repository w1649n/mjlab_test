import numpy as np

from MPC_Controller.common.DesiredStateCommand import DesiredStateCommand
from MPC_Controller.FSM_states.ControlFSMData import ControlFSMData
from MPC_Controller.Parameters import Parameters
from MPC_Controller.common.Quadruped import Quadruped, RobotType
from MPC_Controller.common.LegController import LegController
from MPC_Controller.common.StateEstimator import StateEstimator
from MPC_Controller.convex_MPC.ConvexMPCLocomotion import ConvexMPCLocomotion

class RobotRunnerMin:
    def __init__(self):
        pass


    def init(
        self,
        robotType: RobotType,
        body_mass=None,
        foot_landing_height=None,
        iterations_between_mpc=2,
        enable_stand_mode=False,
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
        stand_foot_search_rate=0.05,
        stand_foot_search_depth=0.08,
        stand_mpc_xy_position_weight=5.0,
        stand_mpc_vxy_weight=3.0,
        stand_kp_cartesian=(80.0, 80.0, 30.0),
        stand_kd_cartesian=(8.0, 8.0, 8.0),
        stand_hipx_kp=12.0,
        stand_hipx_kd=1.0,
        gait_period=None,
        symmetric_residuals=False,
        stop_reposition_steps=0,
    ):
        """
        Initializes the robot model, state estimator, leg controller,
        robot data, and any control logic specific data.
        """
        self.robotType = robotType

        if isinstance(iterations_between_mpc, (bool, np.bool_)) or not isinstance(
            iterations_between_mpc, (int, np.integer)
        ):
            raise ValueError("iterations_between_mpc must be a positive integer")
        if iterations_between_mpc <= 0:
            raise ValueError("iterations_between_mpc must be a positive integer")

        # Keep the gait clock and the C++ MPC solver on one explicit cadence.
        # The legacy 27 ms expression was silently truncated to two controller
        # ticks at dt=10 ms, producing a 5 Hz trot instead of the intended
        # roughly 3--4 Hz gait.
        self.cMPC = ConvexMPCLocomotion(
            Parameters.controller_dt, int(iterations_between_mpc), gait_period=gait_period
        )
        if type(stop_reposition_steps) is not int or stop_reposition_steps not in (0, 2):
            raise ValueError("stop_reposition_steps must be 0 or 2")
        self.cMPC.symmetric_residuals = bool(symmetric_residuals)
        self.cMPC.stop_reposition_steps = stop_reposition_steps
        self.cMPC.configureStandMode(
            enabled=enable_stand_mode,
            enter_linear_speed=stand_enter_linear_speed,
            enter_yaw_rate=stand_enter_yaw_rate,
            exit_linear_speed=stand_exit_linear_speed,
            exit_yaw_rate=stand_exit_yaw_rate,
            command_hold_time=stand_command_hold_time,
            contact_hold_time=stand_contact_hold_time,
            contact_loss_time=stand_contact_loss_time,
            arm_timeout=stand_arm_timeout,
            max_linear_speed=stand_max_linear_speed,
            max_vertical_speed=stand_max_vertical_speed,
            max_yaw_rate=stand_max_yaw_rate,
            max_roll_pitch_rate=stand_max_roll_pitch_rate,
            foot_search_rate=stand_foot_search_rate,
            foot_search_depth=stand_foot_search_depth,
            mpc_xy_position_weight=stand_mpc_xy_position_weight,
            mpc_vxy_weight=stand_mpc_vxy_weight,
            kp_cartesian=stand_kp_cartesian,
            kd_cartesian=stand_kd_cartesian,
            hipx_kp=stand_hipx_kp,
            hipx_kd=stand_hipx_kd,
        )

        # init quadruped
        if self.robotType in RobotType:
            self._quadruped = Quadruped(self.robotType)
        else:
            raise Exception("Invalid RobotType")

        # A simulator integration may carry a payload or use a collision model
        # whose total mass differs from the legacy URDF. Configure the SRBD
        # model before constructing the QP solver so gravity compensation is
        # consistent with the simulated plant.
        if body_mass is not None:
            body_mass = float(body_mass)
            if not np.isfinite(body_mass) or body_mass <= 0.0:
                raise ValueError("body_mass must be positive and finite")
            self._quadruped._bodyMass = body_mass

        if foot_landing_height is not None:
            self.cMPC.setFootLandingHeight(foot_landing_height)

        # init leg controller
        self._legController = LegController(self._quadruped)

        # init state estimator
        self._stateEstimator = StateEstimator(self._quadruped)

        # init desired state command
        self._desiredStateCommand = DesiredStateCommand()
        
        # init controller data
        self.data = ControlFSMData()
        self.data._quadruped = self._quadruped
        self.data._stateEstimator = self._stateEstimator
        self.data._legController = self._legController
        self.data._desiredStateCommand = self._desiredStateCommand

        # init convex MPC controller
        self.cMPC.initialize(self.data)

    def reset(self):
        # An episode boundary must not carry a command or a feed-forward force
        # into the next rollout.  ``run`` also clears commands before every
        # control tick, but clearing them here makes reset itself safe (for
        # example when the simulator applies the cached target once before the
        # next policy action is processed).
        self._legController.zeroCommand()
        self._desiredStateCommand.reset()
        self._stateEstimator.reset()
        self.cMPC.initialize(self.data)

    def run(
        self,
        dof_states,
        body_states,
        commands,
        foot_placement_offsets=None,
        foot_contacts=None,
        terrain_height_samples=None,
    ):
        """
        Runs the overall robot control system by calling each of the major components
        to run each of their respective steps.
        """
        # Update desired commands
        self._desiredStateCommand.updateCommand(commands)

        # Update the joint states
        self._legController.updateData(dof_states)
        self._legController.zeroCommand()
        # self._legController.setEnable(True)

        # Update robot states
        self._stateEstimator.update(body_states)

        # Foot-placement residuals are a separate control channel.  They must
        # not be appended to ``commands`` because DesiredStateCommand reserves
        # command elements 3+ for MPC cost weights.
        if foot_placement_offsets is not None:
            self.cMPC.setFootPlacementOffsets(foot_placement_offsets)
        # Clear missing input explicitly; stale ``True`` contacts must never
        # satisfy a later standing transition.
        self.cMPC.setFootContacts(foot_contacts)
        self.cMPC.setTerrainHeightSamples(terrain_height_samples)
        
        # Run the Control FSM code
        self.cMPC.run(self.data)

        # Sets the leg controller commands for the robot
        legTorques = self._legController.updateCommand()

        return legTorques # numpy (12,) float32
