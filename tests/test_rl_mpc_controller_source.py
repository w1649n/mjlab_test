"""Controller source selection must not depend on a sibling checkout."""

from pathlib import Path

from mjlab.tasks.rl_mpc.controller import backend


def test_defaults_to_bundled_controller(monkeypatch):
  monkeypatch.delenv(backend.CONTROLLER_ROOT_ENV, raising=False)
  root = backend.resolve_controller_root()
  assert root is not None
  assert root == Path(__file__).resolve().parents[1] / "thirdparty/rl-mpc-locomotion"
  assert (root / "setup.py").is_file()
  assert (root / "RL_Environment/cfg/task/G23.yaml").is_file()


def test_explicit_controller_overrides_environment(monkeypatch, tmp_path):
  explicit = tmp_path / "explicit"
  environment = tmp_path / "environment"
  for root in (explicit, environment):
    (root / "MPC_Controller").mkdir(parents=True)
  monkeypatch.setenv(backend.CONTROLLER_ROOT_ENV, str(environment))
  assert backend.resolve_controller_root() == environment
  assert backend.resolve_controller_root(str(explicit)) == explicit
