"""Whole-state history shared by actor and critic, sampled once per policy step."""

import torch

FRAME_DIM = 75
HISTORY_LENGTH = 5
OBS_DIM = FRAME_DIM * HISTORY_LENGTH


class MpcObservationHistory:
  def __init__(self, num_envs: int, device: str | torch.device):
    self.frames = torch.zeros((num_envs, HISTORY_LENGTH, FRAME_DIM), device=device)
    self.last_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    self.frames[env_ids] = 0
    self.last_step[env_ids] = -1

  def update(self, frame: torch.Tensor, step: int) -> torch.Tensor:
    if frame.shape != self.frames[:, 0].shape:
      raise ValueError("MPC observation frame must contain 75 values per environment")
    if not torch.isfinite(frame).all():
      raise ValueError("Non-finite MPC observation frame")
    fresh = self.last_step == -1
    advance = (self.last_step != step) & ~fresh
    self.frames[advance, 1:] = self.frames[advance, :-1].clone()
    self.frames[advance, 0] = frame[advance]
    # Match the reference reset: repeat the first observation into every slot.
    self.frames[fresh] = frame[fresh, None, :].expand(-1, HISTORY_LENGTH, -1)
    self.last_step[:] = step
    return self.frames.flatten(start_dim=1)
