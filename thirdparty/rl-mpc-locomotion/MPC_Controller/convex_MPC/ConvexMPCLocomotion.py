import math
import time
import sys
from enum import IntEnum

import numpy as np
# import MPC_Controller.convex_MPC.mpc_osqp as mpc
from MPC_Controller.common.Quadruped import RobotType
from MPC_Controller.Parameters import Parameters
from MPC_Controller.convex_MPC.Gait import (
    OffsetDurationGait,
    StandingGait,
    StoppingGait,
)
from MPC_Controller.common.DesiredStateCommand import DesiredStateCommand
from MPC_Controller.FSM_states.ControlFSMData import ControlFSMData
from MPC_Controller.common.FootSwingTrajectory import FootSwingTrajectory
from MPC_Controller.utils import CASTING, NUM_LEGS, DTYPE, getSideSign
from MPC_Controller.math_utils.orientation_tools import coordinateRotation, CoordinateAxis
from MPC_Controller.Logger import Logger

try:
    import mpc_osqp as mpc
except:
    print("You need to install 'rl-mpc-locomotion'")
    print("Run 'pip install -e .' in this repo")
    sys.exit()


class LocomotionMode(IntEnum):
    """Controller-local gait mode exposed to the residual policy."""

    TROT = 0
    STOPPING = 1
    STAND = 2

class ConvexMPCLocomotion:
    def __init__(self, _dt:float, _iterationsBetweenMPC:int, gait_period=None):
        self.iterationsBetweenMPC = int(_iterationsBetweenMPC)
        self.horizonLength = 10 # a fixed number for all mpc gait
        self.dt = _dt
        
        self.trotting = OffsetDurationGait(10, 
                            np.array([0, 5, 5, 0], dtype=DTYPE), 
                            np.array([5, 5, 5, 5], dtype=DTYPE), "Trotting",
                            cycle_segments=(None if gait_period is None else
                                            gait_period / (self.dt * self.iterationsBetweenMPC)))
        self.standing = StandingGait(self.horizonLength)
        self.stopping = StoppingGait(
            self.horizonLength,
            swingSegments=self.trotting.getCurrentSwingTime(1.0, 0),
        )
        
        self.bounding = OffsetDurationGait(10,
                            np.array([5, 5, 0, 0], dtype=DTYPE), 
                            np.array([4, 4, 4, 4], dtype=DTYPE), "Bounding")
        
        self.pronking = OffsetDurationGait(10,
                            np.array([0, 0, 0, 0], dtype=DTYPE), 
                            np.array([4, 4, 4, 4], dtype=DTYPE), "Pronking")

        self.pacing = OffsetDurationGait(10,
                            np.array([5, 0, 5, 0], dtype=DTYPE), 
                            np.array([5, 5, 5, 5], dtype=DTYPE), "Pacing")

        self.galloping = OffsetDurationGait(10,
                            np.array([0, 2, 7, 9], dtype=DTYPE), 
                            np.array([4, 4, 4, 4], dtype=DTYPE), "Galloping")

        self.walking = OffsetDurationGait(10,
                            np.array([0, 3, 5, 8], dtype=DTYPE), 
                            np.array([5, 5, 5, 5], dtype=DTYPE), "Walking")

        self.trotRunning = OffsetDurationGait(10,
                            np.array([0, 5, 5, 0], dtype=DTYPE), 
                            np.array([4, 4, 4, 4], dtype=DTYPE), "Trot Running")

        self.dtMPC = self.dt * self.iterationsBetweenMPC
        self.default_iterations_between_mpc = self.iterationsBetweenMPC
        print("[Convex MPC] dt: %.3f iterations: %d, dtMPC: %.3f" % (self.dt, self.iterationsBetweenMPC, self.dtMPC))
        
        self.firstSwing:list = None
        self.firstRun = True
        self.iterationCounter = 0
        self.pFoot = np.zeros((4,3,1), dtype=DTYPE)
        self.f_ff = np.zeros((4,3,1), dtype=DTYPE)
        # Policy telemetry: nominal touchdown XY relative to CoM, before residual.
        self.foothold_heuristic = np.zeros(8, dtype=DTYPE)
        self.policy_contact_state = np.ones(4, dtype=DTYPE)

        self.foot_positions = np.zeros((4,3,1), dtype=DTYPE)

        self.current_gait = 0
        self._x_vel_des = 0.0
        self._y_vel_des = 0.0
        self._yaw_turn_rate = 0.0

        self._roll_des = 0.0
        self._pitch_des = 0.0

        self.footSwingTrajectories = [FootSwingTrajectory() for _ in range(4)]
        self.swingTimes = np.zeros((4,1), dtype=DTYPE)
        self.swingTimeRemaining = [0.0 for _ in range(4)]

        self.Kp = np.array([700, 0, 0, 0, 700, 0, 0, 0, 150], dtype=DTYPE).reshape((3,3))
        self.Kd = np.array([7, 0, 0, 0, 7, 0, 0, 0, 7], dtype=DTYPE).reshape((3,3))
        self.Kp_stance = np.zeros_like(self.Kp)
        self.Kd_stance = self.Kd
        self.KpJoint_stance = np.zeros((3, 3), dtype=DTYPE)
        self.KdJoint_stance = np.identity(3, dtype=DTYPE) * 0.2
        # A stationary four-contact stance needs a restoring target.  Keep
        # these gains separate from ``Kp_stance`` so trotting and STOPPING
        # retain their original force-dominant stance controller.
        self.Kp_stand = np.diag(np.array([80.0, 80.0, 30.0], dtype=DTYPE))
        self.Kd_stand = np.diag(np.array([8.0, 8.0, 8.0], dtype=DTYPE))
        self.KpJoint_stand = np.diag(
            np.array([12.0, 0.0, 0.0], dtype=DTYPE)
        )
        self.KdJoint_stand = np.diag(
            np.array([1.0, 0.2, 0.2], dtype=DTYPE)
        )

        self.logger = Logger("logs/")
        self.last_mpc_inputs = None
        self.last_gait_inputs = None
        # Planar residual foothold commands use leg order FL, FR, HL, HR and
        # live in the controller's yaw-aligned frame.  Pending values may
        # change every policy step; the per-leg latched value stays fixed for
        # the duration of a swing.
        self._pending_foot_placement_offsets = np.zeros((4, 2), dtype=DTYPE)
        self._latched_foot_placement_offsets = np.zeros((4, 2), dtype=DTYPE)
        # Height of the kinematic foot point at touchdown. Point-foot models
        # keep the legacy -3 mm penetration target; robots with a spherical
        # collision foot should override this to radius minus penetration.
        self._foot_landing_height = -0.003
        # Optional local-world terrain samples ``[x, y, surface_z]`` supplied
        # by simulator integrations.  The legacy standalone controller leaves
        # this unset and retains its fixed-plane touchdown behavior.
        self._terrain_height_samples = None
        # Standing is opt-in so existing users retain the original continuously
        # cycling gait.  The mjlab RL-MPC task enables it explicitly.
        self._stand_mode_enabled = False
        self.symmetric_residuals = False
        self.stop_reposition_steps = 0
        self._symmetric_cycle = -1
        self._symmetric_offsets = np.zeros((4, 2), dtype=DTYPE)
        self._home_foot_positions = np.zeros((4, 3, 1), dtype=DTYPE)
        self._stop_steps_remaining = 0
        self._last_swing_diagonal = None
        self._stop_step_active = False
        self._stop_step_contacts = 0
        self._stand_enter_linear_speed = 0.03
        self._stand_enter_yaw_rate = 0.05
        self._stand_exit_linear_speed = 0.08
        self._stand_exit_yaw_rate = 0.10
        self._stand_command_hold_ticks = max(1, int(math.ceil(0.20 / self.dt)))
        self._stand_contact_hold_ticks = max(1, int(math.ceil(0.05 / self.dt)))
        self._stopping_contact_on_hold_ticks = 1
        self._stand_contact_loss_ticks = max(1, int(math.ceil(0.05 / self.dt)))
        self._stand_arm_timeout_ticks = max(1, int(math.ceil(0.20 / self.dt)))
        self._stand_max_linear_speed = 0.10
        self._stand_max_vertical_speed = 0.10
        self._stand_max_yaw_rate = 0.20
        self._stand_max_roll_pitch_rate = 0.20
        self._stand_foot_search_rate = 0.05
        self._stand_foot_search_depth = 0.08
        # STAND-only MPC damping.  The G23 walking weights intentionally keep
        # planar velocity authority low; standing needs stronger damping but
        # must not change the policy's TROT dynamics.
        self._stand_mpc_vxy_weight = 3.0
        self._stand_mpc_xy_position_weight = 5.0
        self._stand_com_hold_xy = np.zeros(2, dtype=DTYPE)
        self._stand_com_target_valid = False
        self._stand_foot_hold_targets_world = np.zeros((4, 3, 1), dtype=DTYPE)
        self._stand_foot_targets_valid = False
        self._stand_foot_latch_requested = False
        self.gait_mode = LocomotionMode.TROT
        self._stand_command_ticks = 0
        self._stand_contact_counter = 0
        self._stand_contact_loss_counter = 0
        self._stand_arm_ticks = 0
        self._stand_contact_confirmed = False
        self._stand_settling = False
        self._stopping_swing_legs = np.zeros(4, dtype=bool)
        self._stopping_swing_progress = np.zeros(4, dtype=DTYPE)
        self._stopping_search_depth = np.zeros(4, dtype=DTYPE)
        self._stopping_contacts = np.ones(4, dtype=bool)
        self._stopping_contact_on_counters = np.zeros(4, dtype=np.int64)
        self._stopping_contact_off_counters = np.zeros(4, dtype=np.int64)
        self._last_stopping_contact_row = None
        self._trot_iteration = 0
        self._ticks_since_mpc_update = 0
        self._foot_contacts = None
        self._hold_feet_requested = False
        self._active_gait_name = "Trotting"
        self._last_valid_f_ff = np.zeros((4, 3, 1), dtype=DTYPE)
        self._last_valid_mpc_contact_mask = None
        self._has_valid_mpc_solution = False
        self._mpc_solver_failure_count = 0
        self._mpc_solver_consecutive_failures = 0
        self._mpc_solver_failure_limit = 5
        self._last_mpc_solve_ok = True
        self._last_mpc_solver_error = None
        self._last_mpc_solver_mode = None
 
    def initialize(self, data:ControlFSMData):
        if Parameters.cmpc_alpha > 1e-4:
            print("Alpha was set too high (" + str(Parameters.cmpc_alpha) + ") adjust to 1e-5\n")
            Parameters.cmpc_alpha = 1e-5

        if Parameters.cmpc_enable_log:
            # flush last log
            if not self.logger.is_empty():
                self.logger.flush_logging()
            # start new logs
            self.logger.start_logging()

        self.iterationCounter = 0
        # ``run`` increments the gait clock before calling
        # ``updateMPCIfNeeded``.  Without this explicit edge flag, the first
        # post-reset tick would reuse the old episode's ``f_ff`` (or zeros)
        # until the normal MPC cadence fires.
        self._mpc_update_required = True
        formulation = mpc.SYNCAI if Parameters.mpc_backend == "syncai" else mpc.NATIVE
        self._cpp_mpc = mpc.ConvexMpc(data._quadruped._bodyMass, 
                            list(data._quadruped._bodyInertia),
                            NUM_LEGS,
                            self.horizonLength,
                            self.dtMPC,
                            Parameters.cmpc_alpha,
                            mpc.QPOASES,
                            formulation)

        self._x_vel_des = 0.0
        self._y_vel_des = 0.0
        self._yaw_turn_rate = 0.0
        self._roll_des = 0.0
        self._pitch_des = 0.0
        self.firstSwing = [True for _ in range(4)]
        self.firstRun = True
        self.current_gait = 0
        self.pFoot.fill(0)
        self.f_ff.fill(0)
        self._last_valid_f_ff.fill(0)
        self.foothold_heuristic.fill(0)
        self.policy_contact_state.fill(1)
        self.foot_positions.fill(0)
        self.swingTimes.fill(0)
        self.swingTimeRemaining = [0.0 for _ in range(4)]
        # Position, velocity, acceleration, and endpoint buffers all belong to
        # one swing.  Recreate the tiny trajectory objects rather than leaving
        # any private buffer from the previous episode live.
        self.footSwingTrajectories = [FootSwingTrajectory() for _ in range(4)]
        self.last_mpc_inputs = None
        self.last_gait_inputs = None
        self._pending_foot_placement_offsets.fill(0)
        self._latched_foot_placement_offsets.fill(0)
        self._terrain_height_samples = None
        self._symmetric_cycle = -1
        self._symmetric_offsets.fill(0)
        self._home_foot_positions.fill(0)
        self._last_swing_diagonal = None
        self._stop_steps_remaining = 0
        self._stop_step_active = False
        self._stop_step_contacts = 0
        self._stand_foot_hold_targets_world.fill(0)
        self._stand_foot_targets_valid = False
        self._stand_com_hold_xy.fill(0)
        self._stand_com_target_valid = False
        self.gait_mode = (
            LocomotionMode.STAND
            if self._stand_mode_enabled
            else LocomotionMode.TROT
        )
        self._stand_foot_latch_requested = (
            self.gait_mode == LocomotionMode.STAND
        )
        self._stand_command_ticks = 0
        self._stand_contact_counter = 0
        self._stand_contact_loss_counter = 0
        self._stand_arm_ticks = 0
        self._stand_contact_confirmed = False
        self._stand_settling = False
        self._stopping_swing_legs.fill(False)
        self._stopping_swing_progress.fill(0)
        self._stopping_search_depth.fill(0)
        self._stopping_contacts.fill(True)
        self._stopping_contact_on_counters.fill(0)
        self._stopping_contact_off_counters.fill(0)
        self._last_stopping_contact_row = None
        self._trot_iteration = 0
        self._ticks_since_mpc_update = 0
        self._foot_contacts = None
        self._hold_feet_requested = False
        self._last_valid_mpc_contact_mask = None
        self._has_valid_mpc_solution = False
        self._mpc_solver_failure_count = 0
        self._mpc_solver_consecutive_failures = 0
        self._last_mpc_solve_ok = True
        self._last_mpc_solver_error = None
        self._last_mpc_solver_mode = None
        self._active_gait_name = (
            "Standing" if self.gait_mode == LocomotionMode.STAND else "Trotting"
        )

    def configureStandMode(
        self,
        enabled=False,
        enter_linear_speed=0.03,
        enter_yaw_rate=0.05,
        exit_linear_speed=0.08,
        exit_yaw_rate=0.10,
        command_hold_time=0.20,
        contact_hold_time=0.05,
        contact_loss_time=0.05,
        arm_timeout=0.20,
        max_linear_speed=0.10,
        max_vertical_speed=0.10,
        max_yaw_rate=0.20,
        max_roll_pitch_rate=0.20,
        foot_search_rate=0.05,
        foot_search_depth=0.08,
        mpc_xy_position_weight=5.0,
        mpc_vxy_weight=3.0,
        kp_cartesian=(80.0, 80.0, 30.0),
        kd_cartesian=(8.0, 8.0, 8.0),
        hipx_kp=12.0,
        hipx_kd=1.0,
    ):
        """Configure command hysteresis and the safe all-contact stand gate."""
        values = {
            "enter_linear_speed": enter_linear_speed,
            "enter_yaw_rate": enter_yaw_rate,
            "exit_linear_speed": exit_linear_speed,
            "exit_yaw_rate": exit_yaw_rate,
            "command_hold_time": command_hold_time,
            "contact_hold_time": contact_hold_time,
            "contact_loss_time": contact_loss_time,
            "arm_timeout": arm_timeout,
            "max_linear_speed": max_linear_speed,
            "max_vertical_speed": max_vertical_speed,
            "max_yaw_rate": max_yaw_rate,
            "max_roll_pitch_rate": max_roll_pitch_rate,
            "foot_search_rate": foot_search_rate,
            "foot_search_depth": foot_search_depth,
            "mpc_xy_position_weight": mpc_xy_position_weight,
            "mpc_vxy_weight": mpc_vxy_weight,
            "hipx_kp": hipx_kp,
            "hipx_kd": hipx_kd,
        }
        values = {name: float(value) for name, value in values.items()}
        if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("stand-mode thresholds must be finite and non-negative")
        kp_cartesian = np.asarray(kp_cartesian, dtype=DTYPE)
        kd_cartesian = np.asarray(kd_cartesian, dtype=DTYPE)
        if kp_cartesian.shape != (3,) or kd_cartesian.shape != (3,):
            raise ValueError("stand Cartesian gains must have shape (3,)")
        if not np.all(np.isfinite(kp_cartesian)) or np.any(kp_cartesian < 0.0):
            raise ValueError("stand Cartesian Kp gains must be finite and non-negative")
        if not np.all(np.isfinite(kd_cartesian)) or np.any(kd_cartesian < 0.0):
            raise ValueError("stand Cartesian Kd gains must be finite and non-negative")
        if values["exit_linear_speed"] <= values["enter_linear_speed"]:
            raise ValueError("stand exit linear speed must exceed enter linear speed")
        if values["exit_yaw_rate"] <= values["enter_yaw_rate"]:
            raise ValueError("stand exit yaw rate must exceed enter yaw rate")

        self._stand_mode_enabled = bool(enabled)
        self._stand_enter_linear_speed = values["enter_linear_speed"]
        self._stand_enter_yaw_rate = values["enter_yaw_rate"]
        self._stand_exit_linear_speed = values["exit_linear_speed"]
        self._stand_exit_yaw_rate = values["exit_yaw_rate"]
        self._stand_command_hold_ticks = max(
            1, int(math.ceil(values["command_hold_time"] / self.dt))
        )
        self._stand_contact_hold_ticks = max(
            1, int(math.ceil(values["contact_hold_time"] / self.dt))
        )
        # A measured touchdown enters the force schedule immediately. The
        # active trajectory stays recoverable and the complete STAND switch
        # still requires the longer all-feet hold below. Delaying this force
        # transfer destabilizes the diagonal support pair on uneven ground.
        self._stopping_contact_on_hold_ticks = 1
        self._stand_contact_loss_ticks = max(
            1, int(math.ceil(values["contact_loss_time"] / self.dt))
        )
        self._stand_arm_timeout_ticks = max(
            1, int(math.ceil(values["arm_timeout"] / self.dt))
        )
        self._stand_max_linear_speed = values["max_linear_speed"]
        self._stand_max_vertical_speed = values["max_vertical_speed"]
        self._stand_max_yaw_rate = values["max_yaw_rate"]
        self._stand_max_roll_pitch_rate = values["max_roll_pitch_rate"]
        self._stand_foot_search_rate = values["foot_search_rate"]
        self._stand_foot_search_depth = values["foot_search_depth"]
        self._stand_mpc_xy_position_weight = values["mpc_xy_position_weight"]
        self._stand_mpc_vxy_weight = values["mpc_vxy_weight"]
        self.Kp_stand = np.diag(kp_cartesian)
        self.Kd_stand = np.diag(kd_cartesian)
        self.KpJoint_stand = np.diag(
            np.array([values["hipx_kp"], 0.0, 0.0], dtype=DTYPE)
        )
        self.KdJoint_stand = np.diag(
            np.array([values["hipx_kd"], 0.2, 0.2], dtype=DTYPE)
        )

    @property
    def stand_ready(self):
        """Whether STAND has been armed by continuous four-foot contact."""
        return (
            self.gait_mode == LocomotionMode.STAND
            and self._stand_contact_confirmed
        )

    @property
    def mpc_solver_failure_count(self):
        """Number of invalid MPC solves since the most recent initialize."""
        return self._mpc_solver_failure_count

    @property
    def mpc_solver_consecutive_failures(self):
        """Current run of invalid MPC solves."""
        return self._mpc_solver_consecutive_failures

    def setFootContacts(self, contacts):
        """Set current FL/FR/HL/HR ground contacts used by the stop gate."""
        if contacts is None:
            self._foot_contacts = None
            return
        contacts = np.asarray(contacts)
        if contacts.shape != (4,):
            raise ValueError("foot contacts must have shape (4,)")
        if not np.all(np.isfinite(contacts)):
            raise ValueError("foot contacts must be finite")
        self._foot_contacts = contacts.astype(bool, copy=True)

    def _transitionToMode(self, mode):
        mode = LocomotionMode(mode)
        if mode == self.gait_mode:
            return

        previous_mode = self.gait_mode
        self.gait_mode = mode
        self._stand_command_ticks = 0
        self._stand_contact_counter = 0
        self._stand_contact_loss_counter = 0
        self._stand_arm_ticks = 0
        self._mpc_update_required = True

        if mode == LocomotionMode.STOPPING:
            self._stop_steps_remaining = (
                self.stop_reposition_steps
                if previous_mode == LocomotionMode.TROT else 0
            )
            self._stop_step_active = False
            self._stop_step_contacts = 0
            # Let a leg already in swing keep its latched landing target, while
            # preventing any later policy sample from moving that target.
            self._pending_foot_placement_offsets.fill(0)
            self._stand_contact_confirmed = False
            self._stand_settling = False
            self._stand_foot_targets_valid = False
            self._stand_foot_latch_requested = False
            self._stand_com_target_valid = False
            last_swing = np.zeros(4, dtype=DTYPE)
            scheduled_stance = np.zeros(4, dtype=bool)
            if self.last_gait_inputs is not None:
                last_swing = np.asarray(
                    self.last_gait_inputs["swing_states"], dtype=DTYPE
                ).reshape(4)
                scheduled_stance[:] = last_swing <= 0.0
            if self._foot_contacts is None:
                self._stopping_contacts.fill(False)
            elif previous_mode == LocomotionMode.TROT:
                # Do not let a one-frame contact-sensor gap remove the planted
                # diagonal exactly when a stop command arrives. Seed the
                # previously scheduled stance and let the normal 50 ms loss
                # debounce below decide whether either foot is truly airborne.
                self._stopping_contacts[:] = self._foot_contacts | scheduled_stance
            else:
                # STAND reaches STOPPING only after its contact-loss debounce
                # has already expired. Preserve that confirmed raw loss rather
                # than reviving a foot from the dormant trot schedule.
                self._stopping_contacts[:] = self._foot_contacts
            self._stopping_swing_legs[:] = last_swing > 0.0
            self._stopping_swing_legs |= ~self._stopping_contacts
            self._stopping_swing_progress[:] = last_swing
            self._stopping_search_depth.fill(0)
            self._stopping_contact_on_counters.fill(0)
            self._stopping_contact_off_counters.fill(0)
            self._last_stopping_contact_row = None
            self._stopping_swing_progress[
                self._stopping_swing_legs
                & (self._stopping_swing_progress <= 0.0)
            ] = 1e-6
        elif mode == LocomotionMode.STAND:
            self._pending_foot_placement_offsets.fill(0)
            self._latched_foot_placement_offsets.fill(0)
            self._trot_iteration = 0
            self.firstSwing = [True for _ in range(4)]
            self.swingTimes.fill(0)
            self.swingTimeRemaining = [0.0 for _ in range(4)]
            self._hold_feet_requested = True
            # ``_transitionToMode`` runs before this tick's kinematics are
            # refreshed. Defer the one-shot world-frame latch until pFoot and
            # the body pose below refer to the same simulator sample.
            self._stand_foot_targets_valid = False
            self._stand_foot_latch_requested = True
            self._stand_com_target_valid = False
            self._stand_contact_confirmed = True
            self._stand_settling = False
            self._stopping_swing_legs.fill(False)
            self._stopping_swing_progress.fill(0)
            self._stopping_search_depth.fill(0)
            self._stopping_contact_on_counters.fill(0)
            self._stopping_contact_off_counters.fill(0)
            self._last_stopping_contact_row = None
        elif mode == LocomotionMode.TROT:
            self._last_swing_diagonal = None
            self._stop_steps_remaining = 0
            self._stop_step_active = False
            self._symmetric_cycle = -1
            # Start every transition to TROT from the same phase and from the
            # measured foot positions, not an arbitrary frozen phase.
            self._pending_foot_placement_offsets.fill(0)
            self._latched_foot_placement_offsets.fill(0)
            self._trot_iteration = 0
            self.firstSwing = [True for _ in range(4)]
            self._hold_feet_requested = True
            self._stand_foot_targets_valid = False
            self._stand_foot_latch_requested = False
            self._stand_com_target_valid = False
            self._stand_contact_confirmed = False
            self._stand_settling = False
            self._stopping_swing_legs.fill(False)
            self._stopping_swing_progress.fill(0)
            self._stopping_search_depth.fill(0)
            self._stopping_contact_on_counters.fill(0)
            self._stopping_contact_off_counters.fill(0)
            self._last_stopping_contact_row = None

    def _bodyIsStableForStand(self, seResult):
        linear_velocity = seResult.vBody.reshape(-1)
        angular_velocity = seResult.omegaBody.reshape(-1)
        return (
            float(np.linalg.norm(linear_velocity[:2]))
            <= self._stand_max_linear_speed
            and abs(float(linear_velocity[2])) <= self._stand_max_vertical_speed
            and abs(float(angular_velocity[2])) <= self._stand_max_yaw_rate
            and float(np.linalg.norm(angular_velocity[:2]))
            <= self._stand_max_roll_pitch_rate
        )

    def _startStopRepositionStep(self):
        """One step is one diagonal swing; two steps place every foot once."""
        diagonal = 0 if self._last_swing_diagonal is None else 1 - self._last_swing_diagonal
        legs = [0, 3] if diagonal == 0 else [1, 2]
        self._last_swing_diagonal = diagonal
        self._stopping_swing_legs.fill(False)
        self._stopping_swing_legs[legs] = True
        self._stopping_swing_progress.fill(0)
        self._stopping_swing_progress[legs] = 1e-6
        self._stopping_search_depth.fill(0)
        self._stopping_contact_on_counters.fill(0)
        self._stopping_contact_off_counters.fill(0)
        self._latched_foot_placement_offsets.fill(0)
        for leg in legs:
            self.firstSwing[leg] = True
        self._stop_step_active = True
        self._stop_step_contacts = 0
        self._mpc_update_required = True

    def _updateSymmetricResiduals(self):
        """Latch bounded residual pairs for a whole walking cycle.

        Sharing a sample across both diagonals is essential: mirroring each
        instantaneous policy output still lets the two swing entries differ.
        Turning/side-stepping allow only 5 mm paired X difference and 5 mm
        Y mirror error, with per-foot residual caps of 1.5 cm X / 0.75 cm Y.
        These constrain RL additions, not the MPC's nominal turning geometry.
        """
        if not self.symmetric_residuals:
            return self._pending_foot_placement_offsets
        period = 2 * self.trotting.getCurrentSwingTime(self.dtMPC, 0)
        cycle = int(max(0, self._trot_iteration - 1) * self.dt / period)
        if cycle != self._symmetric_cycle:
            self._symmetric_cycle = cycle
            straight = abs(self._y_vel_des) <= 0.03 and abs(self._yaw_turn_rate) <= 0.05
            pending = self._pending_foot_placement_offsets
            if not straight:
                pending = np.clip(pending, [-0.015, -0.0075], [0.015, 0.0075])
            for left, right in ((0, 1), (2, 3)):
                x = 0.5 * (pending[left, 0] + pending[right, 0])
                y = 0.5 * (pending[left, 1] - pending[right, 1])
                dx = 0.0 if straight else np.clip(
                    0.5 * (pending[left, 0] - pending[right, 0]), -0.0025, 0.0025)
                dy = 0.0 if straight else np.clip(
                    0.5 * (pending[left, 1] + pending[right, 1]), -0.0025, 0.0025)
                self._symmetric_offsets[left] = [x + dx, y + dy]
                self._symmetric_offsets[right] = [x - dx, -y + dy]
        return self._symmetric_offsets

    def _allFeetDown(self):
        return self._foot_contacts is not None and bool(np.all(self._foot_contacts))

    def _allStoppingFeetDown(self):
        """Require both raw and debounced contact before leaving recovery."""
        return self._allFeetDown() and bool(np.all(self._stopping_contacts))

    def _updateStoppingContacts(self):
        """Debounce each STOPPING contact edge and arm lost-foot recovery."""
        if self._foot_contacts is None:
            raise ValueError(
                "stand mode requires FL/FR/HL/HR foot contacts on every tick"
            )

        contact_gained = self._foot_contacts & ~self._stopping_contacts
        contact_lost = ~self._foot_contacts & self._stopping_contacts
        self._stopping_contact_on_counters[contact_gained] += 1
        self._stopping_contact_on_counters[~contact_gained] = 0
        self._stopping_contact_off_counters[contact_lost] += 1
        self._stopping_contact_off_counters[~contact_lost] = 0

        confirmed_gain = (
            self._stopping_contact_on_counters
            >= self._stopping_contact_on_hold_ticks
        )
        confirmed_loss = (
            self._stopping_contact_off_counters >= self._stand_contact_loss_ticks
        )
        self._stopping_contacts[confirmed_gain] = True
        self._stopping_contacts[confirmed_loss] = False

        newly_airborne = confirmed_loss & ~self._stopping_swing_legs
        self._stopping_swing_legs |= confirmed_loss
        self._stopping_swing_progress[
            newly_airborne & (self._stopping_swing_progress <= 0.0)
        ] = 1e-6

    def _beginStandSettling(self):
        # Keep the recoverable STOPPING gait and its trajectories intact until
        # contact stays valid for the full debounce window.  STAND is then
        # committed atomically by ``_transitionToMode``.
        self._stand_settling = True
        self._stand_contact_counter = 1

    def _cancelStandSettling(self):
        self._stand_settling = False
        self._stand_contact_counter = 0

    def _updateStandMode(self, seResult):
        if not self._stand_mode_enabled:
            self.gait_mode = LocomotionMode.TROT
            return

        command_linear_speed = float(np.linalg.norm([
            self._x_vel_des, self._y_vel_des
        ]))
        command_yaw_rate = abs(float(self._yaw_turn_rate))
        wants_stand = (
            command_linear_speed <= self._stand_enter_linear_speed
            and command_yaw_rate <= self._stand_enter_yaw_rate
        )
        wants_trot = (
            command_linear_speed >= self._stand_exit_linear_speed
            or command_yaw_rate >= self._stand_exit_yaw_rate
        )

        if self.gait_mode == LocomotionMode.STAND:
            all_feet_down = self._allFeetDown()
            body_is_stable = self._bodyIsStableForStand(seResult)
            if not self._stand_contact_confirmed:
                # A non-zero episode command can arrive on the very first
                # simulator tick, before the reset pose has actually loaded
                # all four feet.  Arm the all-contact stance first; otherwise
                # the initial diagonal swing removes support while the robot
                # is still settling onto uneven terrain.
                self._stand_arm_ticks += 1
                self._stand_contact_counter = (
                    self._stand_contact_counter + 1
                    if all_feet_down and body_is_stable
                    else 0
                )
                if self._stand_contact_counter >= self._stand_contact_hold_ticks:
                    self._stand_contact_confirmed = True
                    self._stand_contact_loss_counter = 0
                elif self._stand_arm_ticks >= self._stand_arm_timeout_ticks:
                    # Do not claim all-contact standing indefinitely when the
                    # reset pose failed to establish support. Zero-speed TROT
                    # is the controlled recovery path; environment safety
                    # terminations remain free to reset a failed episode.
                    self._transitionToMode(LocomotionMode.STOPPING)
                if not self._stand_contact_confirmed:
                    return

            if wants_trot:
                # Never start a new diagonal swing while support is already
                # incomplete. Recover the lost foot first.
                self._transitionToMode(
                    LocomotionMode.TROT
                    if all_feet_down
                    else LocomotionMode.STOPPING
                )
                return

            if not all_feet_down:
                self._stand_contact_loss_counter += 1
                if self._stand_contact_loss_counter >= self._stand_contact_loss_ticks:
                    self._transitionToMode(LocomotionMode.STOPPING)
            else:
                self._stand_contact_loss_counter = 0
            return

        if self.gait_mode == LocomotionMode.TROT:
            self._stand_command_ticks = (
                self._stand_command_ticks + 1 if wants_stand else 0
            )
            if self._stand_command_ticks >= self._stand_command_hold_ticks:
                self._transitionToMode(LocomotionMode.STOPPING)
            return

        if wants_trot:
            step_finished = (
                not self._stop_step_active
                or np.all(self._stopping_swing_progress[
                    self._stopping_swing_legs] >= 1.0)
            )
            if step_finished and self._allStoppingFeetDown():
                self._transitionToMode(LocomotionMode.TROT)
            return

        body_is_stable = self._bodyIsStableForStand(seResult)
        all_feet_down = self._allStoppingFeetDown()
        # STOPPING latches the intent to stop.  Commands in the hysteresis
        # band keep that intent; only an exit-threshold command resumes TROT.
        can_settle = (not wants_trot) and body_is_stable and all_feet_down
        if self._stop_steps_remaining:
            if self._stop_step_active:
                landed = np.all(self._stopping_swing_progress[
                    self._stopping_swing_legs] >= 1.0)
                self._stop_step_contacts = (
                    self._stop_step_contacts + 1 if landed and can_settle else 0
                )
                if self._stop_step_contacts < self._stand_contact_hold_ticks:
                    return
                self._stop_steps_remaining -= 1
                self._stop_step_active = False
            if self._stop_steps_remaining:
                if can_settle:
                    self._startStopRepositionStep()
                return
        if self._stand_settling:
            if not can_settle:
                self._cancelStandSettling()
                return
            self._stand_contact_counter += 1
            if self._stand_contact_counter >= self._stand_contact_hold_ticks:
                self._transitionToMode(LocomotionMode.STAND)
            return

        if can_settle:
            self._beginStandSettling()
            if self._stand_contact_counter >= self._stand_contact_hold_ticks:
                self._transitionToMode(LocomotionMode.STAND)

    def setFootPlacementOffsets(self, offsets):
        """Set bounded planar residual footholds for FL, FR, HL, HR.

        Args:
            offsets: Array shaped ``(4, 2)`` in the yaw-aligned frame.  Each
              x/y component is clipped to +/-8 cm.  A leg consumes the newest
              value when it enters swing and holds it until touchdown.
        """
        offsets = np.asarray(offsets, dtype=DTYPE)
        if offsets.shape != (4, 2):
            raise ValueError("foot placement offsets must have shape (4, 2)")
        if not np.all(np.isfinite(offsets)):
            raise ValueError("foot placement offsets must be finite")
        if self._stand_mode_enabled and self.gait_mode != LocomotionMode.TROT:
            self._pending_foot_placement_offsets.fill(0)
            if self.gait_mode == LocomotionMode.STAND:
                self._latched_foot_placement_offsets.fill(0)
            return
        np.copyto(
            self._pending_foot_placement_offsets,
            np.clip(offsets, -0.08, 0.08),
            casting=CASTING,
        )

    def setFootLandingHeight(self, height):
        """Set the terrain-relative height of the kinematic foot point."""
        height = float(height)
        if not np.isfinite(height):
            raise ValueError("foot landing height must be finite")
        self._foot_landing_height = height

    def setTerrainHeightSamples(self, samples):
        """Set local-world terrain points used to choose touchdown height."""
        if samples is None:
            self._terrain_height_samples = None
            return
        samples = np.asarray(samples, dtype=DTYPE)
        if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != 3:
            raise ValueError("terrain height samples must have shape (N, 3)")
        if not np.all(np.isfinite(samples)):
            raise ValueError("terrain height samples must be finite")
        self._terrain_height_samples = samples.copy()

    def _terrainHeightAt(self, xy):
        """Interpolate the scanned terrain surface at a local-world XY point."""
        if self._terrain_height_samples is None:
            return 0.0
        xy = np.asarray(xy, dtype=DTYPE).reshape(2)
        delta = self._terrain_height_samples[:, :2] - xy
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        nearest = int(np.argmin(distance_squared))
        if distance_squared[nearest] <= 1e-8:
            return float(self._terrain_height_samples[nearest, 2])

        count = min(4, self._terrain_height_samples.shape[0])
        indices = np.argpartition(distance_squared, count - 1)[:count]
        weights = 1.0 / np.maximum(distance_squared[indices], 1e-6)
        return float(
            np.dot(weights, self._terrain_height_samples[indices, 2])
            / np.sum(weights)
        )

    def _latchStandTargets(self, base_position_world, body_R_world):
        """Latch one stationary CoM/foothold target on entry to STAND."""
        base_position_world = np.asarray(
            base_position_world, dtype=DTYPE
        ).reshape((3, 1))
        body_R_world = np.asarray(body_R_world, dtype=DTYPE).reshape((3, 3))
        self._stand_com_hold_xy[:] = base_position_world[:2, 0]
        for leg in range(NUM_LEGS):
            target_world = (
                base_position_world
                + body_R_world.T @ self.foot_positions[leg]
            )
            self._stand_foot_hold_targets_world[leg] = target_world
            self.footSwingTrajectories[leg].setHoldPosition(target_world)
        self._stand_com_target_valid = True
        self._stand_foot_targets_valid = True
        self._stand_foot_latch_requested = False

    def _setMpcFailureFallback(self, contact_mask, body_mass):
        """Install finite forces after a transient native solver failure."""
        contact_mask = np.asarray(contact_mask, dtype=bool).reshape(NUM_LEGS)
        can_reuse_last = (
            self._has_valid_mpc_solution
            and self._last_valid_mpc_contact_mask is not None
            and np.array_equal(contact_mask, self._last_valid_mpc_contact_mask)
        )
        if can_reuse_last:
            np.copyto(self.f_ff, self._last_valid_f_ff, casting=CASTING)
            return "last_valid_same_contact_schedule"

        self.f_ff.fill(0)
        active_legs = np.flatnonzero(contact_mask)
        if active_legs.size:
            # The leg controller expects the reaction on the ground (opposite
            # the force on the body), hence negative z for gravity support.
            force_z = -float(body_mass) * 9.81 / active_legs.size
            self.f_ff[active_legs, 2, 0] = force_z
        return "gravity_support_current_contacts"

    def recomputerTiming(self, iterations_per_mpc:int):
        self.iterationsBetweenMPC = iterations_per_mpc
        self.dtMPC = self.dt*iterations_per_mpc

    def __SetupCommand(self, data:ControlFSMData):

        self._body_height = data._quadruped._bodyHeight
        self._x_vel_des = data._desiredStateCommand.x_vel_cmd
        self._y_vel_des = data._desiredStateCommand.y_vel_cmd

        self._yaw_turn_rate = data._desiredStateCommand.yaw_turn_rate

    def solveDenseMPC(self, mpcTable:list, data:ControlFSMData):
        seResult = data._stateEstimator.getResult()
        
        # *MPC Weights
        if data._desiredStateCommand.mpc_weights is None:
            mpc_weight = np.asarray(
                data._quadruped._mpc_weights, dtype=DTYPE
            ).copy()
        else:
            mpc_weight = np.asarray(
                data._desiredStateCommand.mpc_weights, dtype=DTYPE
            ).copy()
        if mpc_weight.shape != (13,) or not np.all(np.isfinite(mpc_weight)):
            raise ValueError("MPC weights must contain 13 finite values")
        if self.gait_mode == LocomotionMode.STAND:
            # These are local copies. Do not mutate the G23/TROT weights that
            # are also part of the learned controller's dynamics.
            mpc_weight[3:5] = np.maximum(
                mpc_weight[3:5], self._stand_mpc_xy_position_weight
            )
            mpc_weight[9:11] = np.maximum(
                mpc_weight[9:11], self._stand_mpc_vxy_weight
            )

        timer = time.time()

        # *Normal Vector of ground
        if Parameters.flat_ground:
            gravity_projection_vec = np.array([0, 0, 1],dtype=DTYPE)
        else:
            gravity_projection_vec = seResult.ground_normal_yaw
        
        # *Google's way of states
        com_roll_pitch_yaw = seResult.rpyBody.flatten()
        # com_roll_pitch_yaw = np.array([seResult.rpyBody[0], seResult.rpyBody[1], 0], dtype=DTYPE)
        com_position = seResult.position.flatten()
        com_angular_velocity = seResult.omegaBody.flatten()
        com_velocity = seResult.vBody.flatten()

        desired_com_position = np.array(
            [com_position[0], com_position[1], self._body_height], dtype=DTYPE
        )
        if self.gait_mode == LocomotionMode.STAND:
            if not self._stand_com_target_valid:
                # This is a defensive path for direct calls to solveDenseMPC;
                # normal ``run`` calls latch both targets before the solve.
                self._stand_com_hold_xy[:] = com_position[:2]
                self._stand_com_target_valid = True
            desired_com_position[:2] = self._stand_com_hold_xy
        desired_com_velocity = np.array([self._x_vel_des, self._y_vel_des, 0], dtype=DTYPE)
        desired_com_roll_pitch_yaw = np.zeros(3, dtype=DTYPE) # walk parallel to the ground
        desired_com_angular_velocity = np.array([0, 0, self._yaw_turn_rate], dtype=DTYPE)
        foot_contact_states = np.asarray(mpcTable, dtype=DTYPE)
        foot_positions_body_frame = np.array(self.foot_positions.flatten(), dtype=DTYPE)
        foot_friction_coeffs = data._quadruped._friction_coeffs

        if Parameters.cmpc_print_states:
            print("------------------------------------------")
            print("COM RPY: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(com_roll_pitch_yaw)))
            print("COM Pos: {: .4f}, {: .4f}, {: .4f}".format(*com_position))
            print("COM Ang: {: .4f}, {: .4f}, {: .4f}".format(*com_angular_velocity))
            print("COM Vel: {: .4f}, {: .4f}, {: .4f}".format(*com_velocity))
            # print("------------------------------------------")
            # print("DES RPY: {: .4f}, {: .4f}, {: .4f}".format(*np.rad2deg(desired_com_roll_pitch_yaw)))
            # print("DES Pos: {: .4f}, {: .4f}, {: .4f}".format(*desired_com_position))
            # print("DES Ang: {: .4f}, {: .4f}, {: .4f}".format(*desired_com_angular_velocity))
            # print("DES Vel: {: .4f}, {: .4f}, {: .4f}".format(*desired_com_velocity))
            print("------------------------------------------")
            print("GND Vec: {: .4f}, {: .4f}, {: .4f}".format(*gravity_projection_vec))

        current_contact_mask = (
            foot_contact_states.reshape((self.horizonLength, NUM_LEGS))[0] > 0.5
        )
        solver_error = None
        solver_fallback = None
        raw_solver_output_size = 0
        fatal_solver_error = False
        workspace_reset_for_mode_change = (
            self._last_mpc_solver_mode is not None
            and self._last_mpc_solver_mode != self.gait_mode
        )
        try:
            if workspace_reset_for_mode_change:
                # STAND changes zero/non-zero QP weights, and therefore the
                # sparse Hessian pattern. OSQP cannot update a workspace across
                # that pattern change; rebuild it before solving the new mode.
                self._cpp_mpc.reset_solver()
            self._last_mpc_solver_mode = self.gait_mode
            raw_contact_forces = self._cpp_mpc.compute_contact_forces(
                mpc_weight, # mpc weights list(12,)
                com_position, # com_position
                com_velocity, # com_velocity
                com_roll_pitch_yaw, # com_roll_pitch_yaw (set yaw to 0.0)
                gravity_projection_vec,  # Normal Vector of ground
                com_angular_velocity, # com_angular_velocity
                foot_contact_states,  # Foot contact states
                foot_positions_body_frame,  # foot_positions_base_frame
                foot_friction_coeffs,  # foot_friction_coeffs
                desired_com_position,  # desired_com_position
                desired_com_velocity,  # desired_com_velocity
                desired_com_roll_pitch_yaw,  # desired_com_roll_pitch_yaw
                desired_com_angular_velocity  # desired_com_angular_velocity
                )
            predicted_contact_forces = np.asarray(
                raw_contact_forces, dtype=DTYPE
            ).reshape(-1)
            raw_solver_output_size = predicted_contact_forces.size
            if raw_solver_output_size < NUM_LEGS * 3:
                raise ValueError(
                    "MPC solver returned "
                    f"{raw_solver_output_size} forces, expected at least "
                    f"{NUM_LEGS * 3}"
                )
            if not np.all(np.isfinite(predicted_contact_forces)):
                raise ValueError("MPC solver returned non-finite contact forces")

            np.copyto(
                self.f_ff,
                predicted_contact_forces[:NUM_LEGS * 3].reshape((NUM_LEGS, 3, 1)),
                casting=CASTING,
            )
            np.copyto(self._last_valid_f_ff, self.f_ff, casting=CASTING)
            self._last_valid_mpc_contact_mask = current_contact_mask.copy()
            self._has_valid_mpc_solution = True
            self._mpc_solver_consecutive_failures = 0
            self._last_mpc_solve_ok = True
            self._last_mpc_solver_error = None
        except Exception as exc:
            self._mpc_solver_failure_count += 1
            self._mpc_solver_consecutive_failures += 1
            self._last_mpc_solve_ok = False
            solver_error = f"{type(exc).__name__}: {exc}"
            self._last_mpc_solver_error = solver_error
            solver_fallback = self._setMpcFailureFallback(
                current_contact_mask, data._quadruped._bodyMass
            )
            predicted_contact_forces = self.f_ff.reshape(-1).copy()
            # OSQP can recover from a transient numerical failure only after
            # discarding its warm-start workspace.
            try:
                self._cpp_mpc.reset_solver()
            except Exception as reset_exc:
                solver_error += (
                    f"; reset failed: {type(reset_exc).__name__}: {reset_exc}"
                )
                self._last_mpc_solver_error = solver_error
            fatal_solver_error = (
                self._mpc_solver_consecutive_failures
                >= self._mpc_solver_failure_limit
            )

        self.last_mpc_inputs = {
            "solver": {
                "horizon_length": self.horizonLength,
                "controller_dt": self.dt,
                "iterations_between_mpc": self.iterationsBetweenMPC,
                "dt_mpc": self.dtMPC,
                "alpha": Parameters.cmpc_alpha,
                "iteration": self.iterationCounter,
                "solve_ok": self._last_mpc_solve_ok,
                "used_fallback": solver_fallback is not None,
                "fallback": solver_fallback,
                "raw_output_size": raw_solver_output_size,
                "failure_count_total": self._mpc_solver_failure_count,
                "failure_count_consecutive": self._mpc_solver_consecutive_failures,
                "failure_limit": self._mpc_solver_failure_limit,
                "error": solver_error,
                "workspace_reset_for_mode_change": workspace_reset_for_mode_change,
            },
            "robot_model": {
                "body_mass": data._quadruped._bodyMass,
                "body_inertia": data._quadruped._bodyInertia,
                "body_height": self._body_height,
                "friction_coeffs": foot_friction_coeffs,
            },
            "environment": {
                "flat_ground": Parameters.flat_ground,
                "ground_normal_vec": gravity_projection_vec,
                "foot_friction_coeffs": foot_friction_coeffs,
            },
            "weights": {
                "qp_weights": mpc_weight,
                "groups": {
                    "roll_pitch_yaw": mpc_weight[0:3],
                    "position_xyz": mpc_weight[3:6],
                    "angular_velocity_xyz": mpc_weight[6:9],
                    "linear_velocity_xyz": mpc_weight[9:12],
                    "gravity_placeholder": mpc_weight[12:13],
                },
            },
            "current_state": {
                "com_roll_pitch_yaw": com_roll_pitch_yaw,
                "com_position": com_position,
                "com_velocity": com_velocity,
                "com_angular_velocity": com_angular_velocity,
            },
            "desired_state": {
                "desired_com_roll_pitch_yaw": desired_com_roll_pitch_yaw,
                "desired_com_position": desired_com_position,
                "desired_com_velocity": desired_com_velocity,
                "desired_com_angular_velocity": desired_com_angular_velocity,
            },
            "contact": {
                "gait": self._active_gait_name,
                "gait_value": Parameters.cmpc_gait.value,
                "locomotion_mode": self.gait_mode.name,
                "locomotion_mode_value": int(self.gait_mode),
                "mpc_table_flat": foot_contact_states,
                "mpc_table_rows_horizon_cols_legs": foot_contact_states.reshape((self.horizonLength, NUM_LEGS)),
                "foot_positions_body_frame": foot_positions_body_frame,
                "foot_positions_rows_legs_cols_xyz": self.foot_positions.reshape((NUM_LEGS, 3)),
            },
            "output": {
                "predicted_contact_forces": predicted_contact_forces[:12],
                "force_feedforward_rows_legs_cols_xyz": self.f_ff.reshape((NUM_LEGS, 3)),
            },
        }

        if Parameters.cmpc_print_update_time:
            print("MPC Update Time %.3f s\n"%(time.time()-timer))
        
        if Parameters.cmpc_enable_log:
            mpc_state_loss = (com_roll_pitch_yaw - desired_com_roll_pitch_yaw).dot(mpc_weight[0:3]) + \
                            (com_position - desired_com_position).dot(mpc_weight[3:6]) + \
                            (com_angular_velocity - desired_com_velocity).dot(mpc_weight[6:9]) + \
                            (com_velocity - desired_com_velocity).dot(mpc_weight[9:12])
                        
            mpc_torque_loss = Parameters.cmpc_alpha * np.sum(predicted_contact_forces[:12])


            log_data_frame = dict(
                COM_RPY = com_roll_pitch_yaw, # COM_RPY
                COM_POS = com_position, # COM_POS
                COM_ANG = com_angular_velocity, # COM_ANG
                COM_VEL = com_velocity, # COM_VEL
                DES_RPY = desired_com_roll_pitch_yaw, # DES_RPY
                DES_POS = desired_com_position, # DES_POS
                DES_ANG = desired_com_angular_velocity, # DES_ANG
                DES_VEL = desired_com_velocity, # DES_VEL
                MPC_GRF = predicted_contact_forces[:12], # MPC_GRF
                MPC_LOS = mpc_state_loss+mpc_torque_loss, # MPC_LOS
                MPC_WEI = mpc_weight, # MPC_WEI
                TIM_STA = self.iterationCounter # TIM_STA
            )
            self.logger.update_logging(log_data_frame)

        if fatal_solver_error:
            raise RuntimeError(
                "MPC solver failed "
                f"{self._mpc_solver_consecutive_failures} consecutive updates; "
                f"last error: {solver_error}"
            )

    def updateMPCIfNeeded(self, mpcTable:list, data:ControlFSMData):
        # Anchor cadence to the most recent solve.  A forced solve on reset or
        # a gait transition must not leave the next update one tick away merely
        # because the unrelated lifetime counter is near a modulo boundary.
        self._ticks_since_mpc_update += 1
        if (
            self._mpc_update_required
            or self._ticks_since_mpc_update >= self.iterationsBetweenMPC
        ):
            self.solveDenseMPC(mpcTable, data)
            self._mpc_update_required = False
            self._ticks_since_mpc_update = 0

    def run(self, data:ControlFSMData):
        # Command Setup
        self.__SetupCommand(data)
        if self._stand_mode_enabled and self._foot_contacts is None:
            raise ValueError(
                "stand mode requires FL/FR/HL/HR foot contacts on every tick"
            )
        gaitNumber = Parameters.cmpc_gait.value
        seResult = data._stateEstimator.getResult()

        if self.gait_mode == LocomotionMode.STOPPING:
            self._updateStoppingContacts()
        self._updateStandMode(seResult)
        if self.gait_mode != LocomotionMode.TROT:
            # STOPPING completes the current swing using a zero-velocity MPC
            # target. STAND uses the same exact zero target with four contacts.
            self._x_vel_des = 0.0
            self._y_vel_des = 0.0
            self._yaw_turn_rate = 0.0

        # StateEstimator later replaces z with terrain-relative body height for
        # the MPC model. Preserve this tick's simulator pose for world-frame
        # stationary-foot kinematics.
        base_position_world = seResult.position.copy()

        # pick gait
        gait = self.trotting
        if gaitNumber == 1:
            gait = self.bounding
        elif gaitNumber == 2:
            gait = self.pronking
        elif gaitNumber == 3:
            gait = self.pacing
        elif gaitNumber == 5:
            gait = self.galloping
        elif gaitNumber == 6:
            gait = self.walking
        elif gaitNumber == 7:
            gait = self.trotRunning

        self.current_gait = gaitNumber
        use_standing_schedule = (
            self._stand_mode_enabled and self.gait_mode == LocomotionMode.STAND
        )
        use_stopping_schedule = (
            self._stand_mode_enabled and self.gait_mode == LocomotionMode.STOPPING
        )
        if use_standing_schedule:
            gait = self.standing
            self._active_gait_name = "Standing"
        elif use_stopping_schedule:
            gait = self.stopping
            self._active_gait_name = "Stopping landing"
        elif self.gait_mode == LocomotionMode.STOPPING:
            self._active_gait_name = "Trotting (stopping)"
        else:
            self._active_gait_name = Parameters.cmpc_gait.name
        # Keep the Python gait clock and the C++ solver horizon on the same
        # dtMPC.  Changing this value after initialize() leaves the already
        # constructed solver on the old timestep.
        self.recomputerTiming(self.default_iterations_between_mpc)
        if use_stopping_schedule:
            contacts = self._stopping_contacts.copy()
            if self._stop_step_active:
                # Ignore lingering liftoff contact while a commanded swing is
                # in flight. Touchdown/search uses measured contact at its end.
                contacts[self._stopping_swing_legs & (
                    self._stopping_swing_progress < 1.0)] = False
            still_landing = self._stopping_swing_legs & ~contacts
            swing_time = self.trotting.getCurrentSwingTime(self.dtMPC, 0)
            self._stopping_swing_progress[still_landing] = np.minimum(
                1.0,
                self._stopping_swing_progress[still_landing]
                + self.dt / swing_time,
            )
            searching = still_landing & (
                self._stopping_swing_progress >= 1.0
            )
            self._stopping_search_depth[searching] = np.minimum(
                self._stand_foot_search_depth,
                self._stopping_search_depth[searching]
                + self.dt * self._stand_foot_search_rate,
            )
            self.stopping.configure(
                self._stopping_swing_legs,
                self._stopping_swing_progress,
                contacts,
            )
            stopping_contact_row = (
                self.stopping.getContactState().reshape(4).astype(bool)
            )
            if (
                self._last_stopping_contact_row is None
                or not np.array_equal(
                    stopping_contact_row, self._last_stopping_contact_row
                )
            ):
                self._mpc_update_required = True
                self._last_stopping_contact_row = stopping_contact_row.copy()
        if self._stand_mode_enabled:
            if not use_standing_schedule and not use_stopping_schedule:
                self._trot_iteration += 1
            gait_iteration = self._trot_iteration
        else:
            gait_iteration = self.iterationCounter
        gait.setIterations(self.iterationsBetweenMPC, gait_iteration)

        for i in range(4):
            self.foot_positions[i] = data._quadruped.getHipLocation(i) + data._legController.datas[i].p
            self.pFoot[i] = self.foot_positions[i] + seResult.position
            # np.copyto(self.pFoot[i], seResult.position + \
                                        # (data._quadruped.getHipLocation(i)+
                                        # data._legController.datas[i].p))
        # self.foot_positions = np.array([self.pFoot[i] - seResult.position for i in range(4)], dtype=DTYPE).reshape((4,3,1))

        # * first time initialization
        if self.firstRun:
            self.firstRun = False
            self._home_foot_positions[:] = self.foot_positions
            data._stateEstimator._init_contact_history(self.foot_positions)
            for i in range(4):
                self.footSwingTrajectories[i].setHeight(0.06)
                self.footSwingTrajectories[i].setHoldPosition(self.pFoot[i])

        if self._hold_feet_requested:
            if not use_standing_schedule:
                for i in range(4):
                    self.footSwingTrajectories[i].setHoldPosition(self.pFoot[i])
            self._hold_feet_requested = False

        if use_standing_schedule and (
            self._stand_foot_latch_requested
            or not self._stand_foot_targets_valid
            or not self._stand_com_target_valid
        ):
            self._latchStandTargets(base_position_world, seResult.rBody)

        if Parameters.flat_ground:
            data._stateEstimator._update_com_position_ground_frame(self.foot_positions)
        else:
            data._stateEstimator._compute_ground_normal_and_com_position(self.foot_positions)
        

        # * foot placement
        for l in range(4):
            self.swingTimes[l] = gait.getCurrentSwingTime(self.dtMPC, l)

        v_des_robot = np.array([self._x_vel_des, self._y_vel_des, 0], dtype=DTYPE).reshape((3,1))
        # interleave_y = [0.08, -0.08, -0.02, 0.02]
        # interleave_gain = -0.2
        # v_abs = math.fabs(v_des_robot[0])

        residuals = self._updateSymmetricResiduals()
        for i in range(4):
            if self.firstSwing[i] and self.gait_mode == LocomotionMode.TROT:
                self._latched_foot_placement_offsets[i] = \
                    residuals[i]
            elif self.firstSwing[i] and self.gait_mode == LocomotionMode.STOPPING:
                self._latched_foot_placement_offsets[i].fill(0)

            if self.firstSwing[i]:
                self.swingTimeRemaining[i] = self.swingTimes[i].item()
            else:
                self.swingTimeRemaining[i] -= self.dt

            # Keep swing clearance independent of the target body height.
            self.footSwingTrajectories[i].setHeight(0.06)
            
            offset = np.array([0, getSideSign(i)*data._quadruped._abadLinkLength, 0], dtype=DTYPE).reshape((3,1))
            pRobotFrame = data._quadruped.getHipLocation(i) + offset
            # pRobotFrame[1] += interleave_y[i] * v_abs * interleave_gain
            stance_time = gait.getCurrentStanceTime(self.dtMPC, i)
            pYawCorrected = coordinateRotation(CoordinateAxis.Z, -self._yaw_turn_rate*stance_time/2) @ pRobotFrame

            Pf = seResult.position + (pYawCorrected + v_des_robot * self.swingTimeRemaining[i])

            p_rel_max = 0.3
            pfx_rel = seResult.vBody[0] * (0.5 + Parameters.cmpc_bonus_swing) * stance_time + \
                      0.03 * (seResult.vBody[0] - v_des_robot[0]) + \
                      (0.5 * seResult.position[2] / 9.81) * (seResult.vBody[1] * self._yaw_turn_rate)
            
            # Match the fore-aft Raibert correction above.  ``stance_time`` is
            # already expressed in seconds; multiplying by ``dtMPC`` a second
            # time made lateral feedback roughly 50x too small at 50 Hz MPC.
            pfy_rel = seResult.vBody[1] * 0.5 * stance_time + \
                      0.03 * (seResult.vBody[1] - v_des_robot[1]) + \
                      (0.5 * seResult.position[2] / 9.81) * (-seResult.vBody[0] * self._yaw_turn_rate)
            
            pfx_rel = min(max(pfx_rel, -p_rel_max), p_rel_max)
            pfy_rel = min(max(pfy_rel, -p_rel_max), p_rel_max)
            Pf[0] += pfx_rel
            Pf[1] += pfy_rel
            if self._stop_step_active:
                # Match the observation's nominal target to the executed home
                # placement, before residuals (which are zero while stopping).
                Pf[:2] = seResult.position[:2] + self._home_foot_positions[i, :2]
            self.foothold_heuristic[i * 2:i * 2 + 2] = (
                Pf[:2, 0] - seResult.position[:2, 0]
            )
            Pf[0] += self._latched_foot_placement_offsets[i, 0]
            Pf[1] += self._latched_foot_placement_offsets[i, 1]
            Pf[2] = self._terrainHeightAt(Pf[:2, 0]) + self._foot_landing_height
            if use_stopping_schedule and self._stopping_swing_legs[i]:
                Pf[2] -= self._stopping_search_depth[i]
            self.footSwingTrajectories[i].setFinalPosition(Pf)

        if use_standing_schedule:
            for i in range(4):
                self.swingTimeRemaining[i] = 0.0

        # calc gait
        self.iterationCounter += 1

        # gait
        contactStates = gait.getContactState()
        swingStates = gait.getSwingState()
        if self.gait_mode == LocomotionMode.TROT:
            # Retain the last nonzero swing through the all-contact exchange
            # tick. STOPPING must continue with the opposite diagonal.
            if swingStates[0, 0] > 0 or swingStates[3, 0] > 0:
                self._last_swing_diagonal = 0
            elif swingStates[1, 0] > 0 or swingStates[2, 0] > 0:
                self._last_swing_diagonal = 1
        self.policy_contact_state[:] = gait.getCurrentContactState()
        mpcTable = gait.getMpcTable()
        self.last_gait_inputs = {
            "gait": self._active_gait_name,
            "gait_value": Parameters.cmpc_gait.value,
            "locomotion_mode": self.gait_mode.name,
            "locomotion_mode_value": int(self.gait_mode),
            "stand_settling": self._stand_settling,
            "stop_steps_remaining": self._stop_steps_remaining,
            "stop_step_active": self._stop_step_active,
            "stand_ready": self.stand_ready,
            "stopping_search_depth": self._stopping_search_depth.copy(),
            "foot_contacts": (
                None if self._foot_contacts is None else self._foot_contacts.copy()
            ),
            "debounced_foot_contacts": self._stopping_contacts.copy(),
            "contact_on_counters": self._stopping_contact_on_counters.copy(),
            "contact_off_counters": self._stopping_contact_off_counters.copy(),
            "contact_states": contactStates,
            "swing_states": swingStates,
            "swing_times": self.swingTimes,
            "swing_time_remaining": self.swingTimeRemaining,
            "pending_foot_placement_offsets": self._pending_foot_placement_offsets.copy(),
            "latched_foot_placement_offsets": self._latched_foot_placement_offsets.copy(),
            "stand_com_hold_xy": self._stand_com_hold_xy.copy(),
            "stand_foot_hold_targets_world": self._stand_foot_hold_targets_world.copy(),
            "stand_targets_valid": (
                self._stand_com_target_valid and self._stand_foot_targets_valid
            ),
            "mpc_table_rows_horizon_cols_legs": np.asarray(mpcTable, dtype=DTYPE).reshape((self.horizonLength, NUM_LEGS)),
        }

        # * update MPC
        self.updateMPCIfNeeded(mpcTable, data)

        se_contactState = np.array([0,0,0,0], dtype=DTYPE).reshape((4,1))

        for foot in range(4):
            contactState = contactStates[foot]
            swingState = swingStates[foot]
            if swingState > 0: #* foot is in swing
                if self.firstSwing[foot]:
                    self.firstSwing[foot] = False
                    self.footSwingTrajectories[foot].setInitialPosition(self.pFoot[foot])

                self.footSwingTrajectories[foot].computeSwingTrajectoryBezier(swingState, self.swingTimes[foot].item())
                pDesFoot = self.footSwingTrajectories[foot].getPosition()
                vDesFoot = self.footSwingTrajectories[foot].getVelocity()

                pDesLeg = (pDesFoot - seResult.position) \
                          - data._quadruped.getHipLocation(foot)
                vDesLeg = (vDesFoot - seResult.vBody)

                # data._legController.commands[foot].pDes = pDesLeg
                # data._legController.commands[foot].vDes = vDesLeg
                # data._legController.commands[foot].kpCartesian = self.Kp
                # data._legController.commands[foot].kdCartesian = self.Kd

                np.copyto(data._legController.commands[foot].pDes, pDesLeg, casting=CASTING)
                np.copyto(data._legController.commands[foot].vDes, vDesLeg, casting=CASTING)
                np.copyto(data._legController.commands[foot].kpCartesian, self.Kp, casting=CASTING)
                np.copyto(data._legController.commands[foot].kdCartesian, self.Kd, casting=CASTING)

            else: #* foot is in stance
                self.firstSwing[foot] = True
                if use_standing_schedule and self._stand_foot_targets_valid:
                    pDesFoot = self._stand_foot_hold_targets_world[foot]
                    pDesFootBody = seResult.rBody @ (
                        pDesFoot - base_position_world
                    )
                    pDesLeg = (
                        pDesFootBody - data._quadruped.getHipLocation(foot)
                    )
                    # A world-fixed foot moves opposite both base translation
                    # and rotation when represented in the body frame.
                    rotational_velocity = np.cross(
                        seResult.omegaBody.reshape(3),
                        pDesFootBody.reshape(3),
                    ).reshape((3, 1))
                    vDesLeg = -seResult.vBody - rotational_velocity
                    stance_kp = self.Kp_stand
                    stance_kd = self.Kd_stand
                    stance_kp_joint = self.KpJoint_stand
                    stance_kd_joint = self.KdJoint_stand
                else:
                    pDesFoot = self.footSwingTrajectories[foot].getPosition()
                    vDesFoot = self.footSwingTrajectories[foot].getVelocity()
                    pDesLeg = (pDesFoot - seResult.position) \
                              - data._quadruped.getHipLocation(foot)
                    vDesLeg = (vDesFoot - seResult.vBody)
                    stance_kp = self.Kp_stance
                    stance_kd = self.Kd_stance
                    stance_kp_joint = self.KpJoint_stance
                    stance_kd_joint = self.KdJoint_stance
                
                # data._legController.commands[foot].pDes = pDesLeg
                # data._legController.commands[foot].vDes = vDesLeg
                # data._legController.commands[foot].kpCartesian = self.Kp_stance
                # data._legController.commands[foot].kdCartesian = self.Kd_stance

                # data._legController.commands[foot].forceFeedForward = self.f_ff[foot]
                # data._legController.commands[foot].kdJoint = np.identity(3, dtype=DTYPE)*0.2

                np.copyto(data._legController.commands[foot].pDes, pDesLeg, casting=CASTING)
                np.copyto(data._legController.commands[foot].vDes, vDesLeg, casting=CASTING)
                np.copyto(data._legController.commands[foot].kpCartesian, stance_kp, casting=CASTING)
                np.copyto(data._legController.commands[foot].kdCartesian, stance_kd, casting=CASTING)
                np.copyto(data._legController.commands[foot].forceFeedForward, self.f_ff[foot], casting=CASTING)
                if use_standing_schedule:
                    # Internal neutral HipX posture hold. This is deliberately
                    # not connected to the RL residual action.
                    data._legController.commands[foot].qDes[0, 0] = 0.0
                    data._legController.commands[foot].qdDes[0, 0] = 0.0
                np.copyto(data._legController.commands[foot].kpJoint, stance_kp_joint, casting=CASTING)
                np.copyto(data._legController.commands[foot].kdJoint, stance_kd_joint, casting=CASTING)

                se_contactState[foot] = contactState

        data._stateEstimator.setContactPhase(se_contactState)
