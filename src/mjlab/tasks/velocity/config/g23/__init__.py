from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  syncai_g23_flat_proprio_env_cfg,
  syncai_g23_flat_stand_history6_env_cfg,
  syncai_g23_rough_env_cfg,
  syncai_g23_rough_proprio_env_cfg,
  syncai_g23_rough_proprio_history6_env_cfg,
  syncai_g23_rough_proprio_history6_stair_goals_env_cfg,
  syncai_g23_rough_proprio_history6_stair_yaw_env_cfg,
  syncai_g23_rough_proprio_history6_stair_yaw_j60_env_cfg,
)
from .rl_cfg import syncai_g23_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-SyncAI-G23-Proprio-History6-Stand",
  env_cfg=syncai_g23_flat_stand_history6_env_cfg(),
  play_env_cfg=syncai_g23_flat_stand_history6_env_cfg(play=True),
  rl_cfg=syncai_g23_ppo_runner_cfg(experiment_name="g23_velocity_proprio_history6"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-SyncAI-G23",
  env_cfg=syncai_g23_rough_env_cfg(),
  play_env_cfg=syncai_g23_rough_env_cfg(play=True),
  rl_cfg=syncai_g23_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-SyncAI-G23-Proprio",
  env_cfg=syncai_g23_rough_proprio_env_cfg(),
  play_env_cfg=syncai_g23_rough_proprio_env_cfg(play=True),
  rl_cfg=syncai_g23_ppo_runner_cfg(experiment_name="g23_velocity_proprio"),
  runner_cls=VelocityOnPolicyRunner,
)

# 2026-09-02 stair-training update: opt-in six-frame actor history. This task
# intentionally uses a new experiment because old 45-D checkpoints are incompatible.
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6",
  env_cfg=syncai_g23_rough_proprio_history6_env_cfg(),
  play_env_cfg=syncai_g23_rough_proprio_history6_env_cfg(play=True),
  rl_cfg=syncai_g23_ppo_runner_cfg(experiment_name="g23_velocity_proprio_history6"),
  runner_cls=VelocityOnPolicyRunner,
)

# 2026-10-15 stair-yaw update: History6 plus a stair heading-drift penalty. It
# shares the History6 experiment so existing runs can be resumed with --load-run.
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-StairYaw",
  env_cfg=syncai_g23_rough_proprio_history6_stair_yaw_env_cfg(),
  play_env_cfg=syncai_g23_rough_proprio_history6_stair_yaw_env_cfg(play=True),
  rl_cfg=syncai_g23_ppo_runner_cfg(experiment_name="g23_velocity_proprio_history6"),
  runner_cls=VelocityOnPolicyRunner,
)

# 2026-10-16 J60 update: StairYaw with heading weight -10, J60 torque-speed limits,
# and a motor heating penalty. Shares the experiment so StairYaw runs can resume.
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-StairYaw-J60",
  env_cfg=syncai_g23_rough_proprio_history6_stair_yaw_j60_env_cfg(),
  play_env_cfg=syncai_g23_rough_proprio_history6_stair_yaw_j60_env_cfg(play=True),
  rl_cfg=syncai_g23_ppo_runner_cfg(experiment_name="g23_velocity_proprio_history6"),
  runner_cls=VelocityOnPolicyRunner,
)

# 2026-10-16 stair-goals update: J60 StairYaw plus heading-hold straight commands,
# 5-25 cm stairs with a command-following curriculum, riser-scaled foot clearance,
# stance settling, an above-rating heating penalty, and left-right symmetry
# augmentation. Shares the experiment so StairYaw and J60 runs can resume.
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-J60-StairGoals",
  env_cfg=syncai_g23_rough_proprio_history6_stair_goals_env_cfg(),
  play_env_cfg=syncai_g23_rough_proprio_history6_stair_goals_env_cfg(play=True),
  # Lower entropy while fine-tuning: the first run's action std grew 0.43 -> 0.59.
  rl_cfg=syncai_g23_ppo_runner_cfg(
    experiment_name="g23_velocity_proprio_history6",
    symmetry=True,
    entropy_coef=0.003,
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-SyncAI-G23-Proprio",
  env_cfg=syncai_g23_flat_proprio_env_cfg(),
  play_env_cfg=syncai_g23_flat_proprio_env_cfg(play=True),
  rl_cfg=syncai_g23_ppo_runner_cfg(experiment_name="g23_velocity_flat_proprio"),
  runner_cls=VelocityOnPolicyRunner,
)
