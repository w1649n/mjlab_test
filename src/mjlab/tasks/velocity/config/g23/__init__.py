from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  syncai_g23_flat_proprio_env_cfg,
  syncai_g23_flat_stand_history6_env_cfg,
  syncai_g23_rough_env_cfg,
  syncai_g23_rough_proprio_env_cfg,
  syncai_g23_rough_proprio_history6_env_cfg,
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

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-SyncAI-G23-Proprio",
  env_cfg=syncai_g23_flat_proprio_env_cfg(),
  play_env_cfg=syncai_g23_flat_proprio_env_cfg(play=True),
  rl_cfg=syncai_g23_ppo_runner_cfg(experiment_name="g23_velocity_flat_proprio"),
  runner_cls=VelocityOnPolicyRunner,
)
