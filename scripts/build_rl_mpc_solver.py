"""Build rl-mpc-locomotion's native extension for the active mjlab Python."""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _default_controller_root() -> Path:
  return Path(__file__).resolve().parents[2] / "rl-mpc-locomotion"


def _find_modern_pybind11_include() -> Path:
  try:
    pybind11 = importlib.import_module("pybind11")
    return Path(pybind11.get_include())
  except ImportError:
    # Torch vendors current pybind11 headers and is already a required mjlab
    # dependency. This avoids modifying the runtime environment just to compile.
    import torch

    include = Path(torch.__file__).resolve().parent / "include"
    if (include / "pybind11" / "pybind11.h").is_file():
      return include
  raise RuntimeError("pybind11 >= 2.13 headers were not found")


def build_solver(controller_root: Path) -> Path:
  controller_root = controller_root.expanduser().resolve()
  if not (controller_root / "setup.py").is_file():
    raise FileNotFoundError(f"No rl-mpc-locomotion setup.py under {controller_root}")

  pybind11_include = _find_modern_pybind11_include()
  build_env = os.environ.copy()
  build_env["RL_MPC_PYBIND11_INCLUDE"] = str(pybind11_include)
  with tempfile.TemporaryDirectory(prefix="mjlab-rlmpc-build-") as build_temp:
    subprocess.run(
      [
        sys.executable,
        "setup.py",
        "build_ext",
        "--inplace",
        "--build-temp",
        build_temp,
      ],
      cwd=controller_root,
      env=build_env,
      check=True,
    )

  suffixes = importlib.machinery.EXTENSION_SUFFIXES
  matches = [
    path for suffix in suffixes for path in controller_root.glob(f"mpc_osqp*{suffix}")
  ]
  if not matches:
    raise RuntimeError(
      "Build completed but no interpreter-compatible mpc_osqp was found"
    )

  solver = max(matches, key=lambda path: path.stat().st_mtime)
  probe = (
    "import sys; "
    f"sys.path.insert(0, {str(controller_root)!r}); "
    "import mpc_osqp; "
    "assert mpc_osqp.TEST == 42; "
    "print(mpc_osqp.__file__)"
  )
  subprocess.run([sys.executable, "-c", probe], check=True)
  return solver


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Rebuild mpc_osqp for the active mjlab Python interpreter."
  )
  parser.add_argument(
    "--controller-root",
    type=Path,
    default=_default_controller_root(),
    help="Path to the rl-mpc-locomotion checkout.",
  )
  args = parser.parse_args()
  solver = build_solver(args.controller_root)
  print(f"[OK] Built and imported {solver}")


if __name__ == "__main__":
  main()
