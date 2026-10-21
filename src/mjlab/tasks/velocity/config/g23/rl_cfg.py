"""RL configuration for SyncAI G23 velocity task."""

from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# 2026-10-16 stair-goals update: left-right mirror augmentation for PPO.
G23_SYMMETRY_FUNC = "mjlab.tasks.velocity.config.g23.symmetry:mirror_g23_velocity"


def _g23_symmetry_cfg() -> dict[str, Any]:
  return {
    "data_augmentation_func": G23_SYMMETRY_FUNC,
    "use_data_augmentation": True,
    "use_mirror_loss": False,
    "mirror_loss_coeff": 0.0,
  }


def syncai_g23_ppo_runner_cfg(
  experiment_name: str = "g23_velocity",
  symmetry: bool = False,
  entropy_coef: float = 0.01,
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for SyncAI G23 velocity task.

  ``symmetry`` adds left-right mirrored samples to every PPO mini-batch.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=entropy_coef,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      symmetry_cfg=_g23_symmetry_cfg() if symmetry else None,
    ),
    experiment_name=experiment_name,
    # Keep raw policy actions and the actor's previous-action observation bounded.
    # This still leaves ample range for G23 joint targets while preventing a rare
    # action outlier from destabilizing the observation normalizer and value loss.
    clip_actions=5.0,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
