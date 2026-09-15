import math

from MPC_Controller.utils import GaitType, FSM_OperatingMode, FSM_StateName

class Parameters:
    cmpc_x_drag = 3.0
    cmpc_bonus_swing = 0.0
    cmpc_alpha = 1e-5

    cmpc_print_solver_time = False
    cmpc_print_update_time = False
    cmpc_print_states = False
    cmpc_enable_log = False


    cmpc_py_solver = 1 # 0 cvxopt, 1 osqp
    cmpc_solver_type = 2 # 0 mit py solver, 1 mit cpp solver, 2 google cpp solver
    cmpc_gait = GaitType.TROT # 1 bound, 2 pronk, 3 pace, 4 stand, else trot

    policy_print_time = False

    bridge_MPC_to_RL = False
    mpc_backend = "native"

    flat_ground = False

    # * [-1, 1] -> [a, b] => [-1, 1] * (b-a)/2 + (b+a)/2
    MPC_param_scale = [4, 4, 4,     # 1-9
                       20, 20, 20,  # 30-70
                       1, 1, 1,     # 0-2
                       1, 1, 1]     # 0-2
    
    MPC_param_const = [5, 5, 5,
                       50,50,50,
                       1, 1, 1,
                       1, 1, 1]

    if bridge_MPC_to_RL:
        control_mode = FSM_StateName.LOCOMOTION
        operatingMode = FSM_OperatingMode.TEST
        FSM_check_safety = False
    else:
        control_mode = FSM_StateName.RECOVERY_STAND # 0 passive, 4 locomotion, 6 recovery stand
        operatingMode = FSM_OperatingMode.NORMAL # 0 no transition and safe check, 1 normal
        FSM_check_safety = True
    
    controller_dt = 0.01 # in sec

    locomotionUnsafe = False # global indicator for switching contorl mode
    # FSM_check_safety = False
    FSM_print_info = False

    @classmethod
    def set_mpc_param_mapping(cls, scale=None, const=None):
        if scale is None and const is None:
            return

        if scale is not None:
            scale = [float(x) for x in scale]
            if len(scale) != 12 or not all(math.isfinite(x) and x >= 0.0 for x in scale):
                raise ValueError("MPC_param_scale must contain 12 finite non-negative values")
            cls.MPC_param_scale = scale

        if const is not None:
            const = [float(x) for x in const]
            if len(const) != 12 or not all(math.isfinite(x) and x >= 0.0 for x in const):
                raise ValueError("MPC_param_const must contain 12 finite non-negative values")
            cls.MPC_param_const = const

    @classmethod
    def set_mpc_backend(cls, backend):
        backend = str(backend).lower()
        if backend not in ("native", "syncai"):
            raise ValueError("MPC backend must be 'native' or 'syncai'")
        cls.mpc_backend = backend
