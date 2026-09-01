"""Script to play RL agent with RSL-RL."""

import os
import select
import sys
import time as _time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from threading import Event, Thread
from typing import Literal, Protocol

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.keys import (
  KEY_A,
  KEY_D,
  KEY_E,
  KEY_M,
  KEY_Q,
  KEY_R,
  KEY_S,
  KEY_SPACE,
  KEY_W,
  KEY_X,
)
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


class _ManualVelocityCommand(Protocol):
  def set_manual_command(
    self, env_idx: int, command: tuple[float, float, float] | None
  ) -> tuple[float, float, float] | None: ...


class _VelocityKeyboardController:
  """Translate terminal key codes into per-environment velocity setpoints."""

  _DELTAS = {
    KEY_W: (1.0, 0.0, 0.0),
    KEY_S: (-1.0, 0.0, 0.0),
    KEY_A: (0.0, 1.0, 0.0),
    KEY_D: (0.0, -1.0, 0.0),
    KEY_Q: (0.0, 0.0, 1.0),
    KEY_E: (0.0, 0.0, -1.0),
  }

  def __init__(
    self,
    term: _ManualVelocityCommand,
    get_env_idx: Callable[[], int],
    step: float = 0.1,
  ) -> None:
    if not isfinite(step) or step <= 0.0:
      raise ValueError(
        f"Terminal command step must be positive and finite, got {step}."
      )
    self._term = term
    self._get_env_idx = get_env_idx
    self._step = step
    self._commands: dict[int, tuple[float, float, float]] = {}

  def handles_key(self, key: int) -> bool:
    return key in self._DELTAS or key in (KEY_M, KEY_SPACE, KEY_X)

  def __call__(self, key: int) -> bool:
    env_idx = self._get_env_idx()
    if key == KEY_M:
      if env_idx in self._commands:
        self._term.set_manual_command(env_idx, None)
        del self._commands[env_idx]
        self._print_random(env_idx)
      else:
        self._set_command(env_idx, (0.0, 0.0, 0.0))
      return True

    if key in (KEY_SPACE, KEY_X):
      self._set_command(env_idx, (0.0, 0.0, 0.0))
      return True

    delta = self._DELTAS.get(key)
    if delta is None:
      return False
    current = self._commands.get(env_idx, (0.0, 0.0, 0.0))
    command = (
      current[0] + self._step * delta[0],
      current[1] + self._step * delta[1],
      current[2] + self._step * delta[2],
    )
    self._set_command(env_idx, command)
    return True

  def _set_command(self, env_idx: int, command: tuple[float, float, float]) -> None:
    applied = self._term.set_manual_command(env_idx, command)
    assert applied is not None
    self._commands[env_idx] = applied
    vx, vy, yaw = applied
    print(
      f"\n[COMMAND] env {env_idx}: MANUAL "
      f"vx={vx:+.2f} m/s, vy={vy:+.2f} m/s, yaw={yaw:+.2f} rad/s"
    )

  @staticmethod
  def _print_random(env_idx: int) -> None:
    print(f"\n[COMMAND] env {env_idx}: RANDOM (sampled a new command)")


def _dispatch_terminal_key(
  key: int,
  keyboard: _VelocityKeyboardController,
  request_callback: Callable[[Callable[[], None]], None],
  request_reset: Callable[[], None],
) -> bool:
  """Queue terminal controls without touching simulation state on the reader thread."""
  if key == KEY_R:
    request_reset()
    print("\n[ACTION] Robot reset requested.")
    return True
  if not keyboard.handles_key(key):
    return False

  def apply_key() -> None:
    keyboard(key)

  request_callback(apply_key)
  return True


class _TerminalKeyReader:
  """Read single keys from a POSIX terminal without blocking the viewer."""

  def __init__(self, on_key: Callable[[int], bool]) -> None:
    self._on_key = on_key
    self._stop = Event()
    self._thread: Thread | None = None
    self._fd: int | None = None
    self._saved_terminal: list | None = None
    self._escape_state = 0
    self.enabled = False

  @staticmethod
  def _decode_key_codes(data: bytes, state: int = 0) -> tuple[list[int], int]:
    keys: list[int] = []
    for value in data:
      if state == 0:
        if value == 0x1B:
          state = 1
        elif 0x61 <= value <= 0x7A:
          keys.append(value - 0x20)
        else:
          keys.append(value)
      elif state == 1:
        if value in (ord("["), ord("O")):
          state = 2
        else:
          state = 0
      elif 0x40 <= value <= 0x7E:
        state = 0
    return keys, state

  @staticmethod
  def _key_codes(data: bytes):
    """Yield keys from one byte block, excluding ANSI escape sequences."""
    keys, _ = _TerminalKeyReader._decode_key_codes(data)
    yield from keys

  def __enter__(self) -> "_TerminalKeyReader":
    if os.name != "posix":
      return self
    import termios
    import tty

    try:
      fd = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
      return self
    if not os.isatty(fd):
      return self

    try:
      saved_terminal = termios.tcgetattr(fd)
    except OSError:
      return self
    try:
      tty.setcbreak(fd)
      self._fd = fd
      self._saved_terminal = saved_terminal
      self._thread = Thread(
        target=self._read_loop, name="terminal-command", daemon=True
      )
      self._thread.start()
    except (OSError, RuntimeError):
      termios.tcsetattr(fd, termios.TCSAFLUSH, saved_terminal)
      self._fd = None
      self._saved_terminal = None
      return self
    self.enabled = True
    return self

  def __exit__(self, *_exc_info) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=0.5)
    if self._fd is not None and self._saved_terminal is not None:
      import termios

      try:
        termios.tcsetattr(self._fd, termios.TCSAFLUSH, self._saved_terminal)
      except (OSError, termios.error) as exc:
        print(f"\n[WARN]: Could not restore terminal settings: {exc}")
    self.enabled = False

  def _read_loop(self) -> None:
    assert self._fd is not None
    while not self._stop.is_set():
      try:
        readable, _, _ = select.select([self._fd], [], [], 0.1)
      except OSError:
        break
      if not readable:
        continue
      try:
        data = os.read(self._fd, 32)
      except OSError:
        break
      if not data:
        break
      keys, self._escape_state = self._decode_key_codes(data, self._escape_state)
      for key in keys:
        try:
          self._on_key(key)
        except Exception as exc:
          print(f"\n[WARN]: Terminal command failed: {exc}")


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  terminal_commands: bool = True
  """Read velocity-command controls from the terminal during play."""
  terminal_command_step: float = 0.1
  """Velocity and yaw increment applied by each terminal key press."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  if cfg.terminal_commands and (
    not isfinite(cfg.terminal_command_step) or cfg.terminal_command_step <= 0.0
  ):
    raise ValueError(
      "terminal_command_step must be positive and finite, "
      f"got {cfg.terminal_command_step}."
    )

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  velocity_command_term: UniformVelocityCommand | None = None
  for command_name in env.command_manager.active_terms:
    command_term = env.command_manager.get_term(command_name)
    if isinstance(command_term, UniformVelocityCommand):
      velocity_command_term = command_term
      break

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    viewer = NativeMujocoViewer(env, policy)
  elif resolved_viewer == "viser":
    viewer = ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager)
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  try:
    if cfg.terminal_commands and velocity_command_term is not None:

      def get_selected_env_idx() -> int:
        native_env_idx = getattr(viewer, "env_idx", None)
        if native_env_idx is not None:
          return int(native_env_idx)
        scene = getattr(viewer, "_scene", None)
        return int(scene.env_idx if scene is not None else viewer.cfg.env_idx)

      keyboard = _VelocityKeyboardController(
        velocity_command_term,
        get_selected_env_idx,
        step=cfg.terminal_command_step,
      )

      def queue_terminal_key(key: int) -> bool:
        return _dispatch_terminal_key(
          key, keyboard, viewer.request_callback, viewer.request_reset
        )

      with _TerminalKeyReader(queue_terminal_key) as terminal:
        if terminal.enabled:
          print(
            "[INFO]: Terminal command controls: M Random/Manual, "
            "W/S forward/backward, A/D left/right, Q/E yaw, "
            "Space zero, R reset."
          )
          print(
            f"[INFO]: Each key changes the setpoint by "
            f"{cfg.terminal_command_step:g}; direction keys automatically enter "
            "Manual mode."
          )
        else:
          print(
            "[WARN]: Terminal commands are unavailable because stdin is not a TTY; "
            "random commands remain active."
          )
        viewer.run()
    else:
      viewer.run()
  finally:
    env.close()


def main():
  maybe_print_top_level_help("play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
