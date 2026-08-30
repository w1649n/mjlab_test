#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CORE_PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

PROJECT_ARG=""
LOG_ROOT_ARG=""
OUTPUT_ROOT_ARG=""
TASK_ARG=""
RUN_ARG=""
CHECKPOINT_ARG=""
OUTPUT_NAME_ARG=""
DEVICE="cpu"
FORCE=false
DRY_RUN=false
NO_METADATA=false
VERBOSE=false
LIST_TASKS=false
ANSWER=""

usage() {
  cat <<'EOF'
互動式匯出 mjlab RSL-RL checkpoint 為 ONNX。

用法：
  ./scripts/export_onnx.sh [選項]

選項：
  --project DIR        指定 mjlab 專案目錄；預設使用目前 mjlab_test
  --log-root DIR       RSL-RL logs 根目錄；預設互動詢問 logs/rsl_rl
  --output-root DIR    ONNX 輸出目錄；預設互動選擇
  --task TASK_ID       指定 task，省略時互動選擇
  --run RUN            指定 run 名稱或 latest，省略時互動選擇
  --checkpoint STEP    指定 checkpoint iteration、model_*.pt 或 latest
  --output-name NAME   指定輸出 ONNX 檔名
  --device DEVICE      建立匯出環境的 device；預設 cpu
  --force              覆寫既有輸出檔
  --no-metadata        不寫入 ONNX metadata
  --verbose            顯示 torch.onnx export 詳細 graph
  --dry-run            只顯示設定，不執行匯出
  --list-tasks         列出 task 和 checkpoint 數量後結束
  -h, --help           顯示此說明

範例：
  ./scripts/export_onnx.sh

  ./scripts/export_onnx.sh \
    --task Mjlab-Velocity-Rough-SyncAI-G23-Proprio \
    --run 2026-08-30_01-05-22 \
    --checkpoint 7900 \
    --output-root ../quadruped_robot_simulation/policy/ppo \
    --output-name policy_g23_mjlab_7900.onnx \
    --force
EOF
}

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

cancel() {
  printf '\n[INFO] 已取消。\n'
  exit 0
}

trap cancel INT TERM

while (($# > 0)); do
  case "$1" in
    --project)
      (($# >= 2)) || die "--project 後面需要一個目錄。"
      PROJECT_ARG=$2
      shift 2
      ;;
    --log-root)
      (($# >= 2)) || die "--log-root 後面需要一個目錄。"
      LOG_ROOT_ARG=$2
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || die "--output-root 後面需要一個目錄。"
      OUTPUT_ROOT_ARG=$2
      shift 2
      ;;
    --task)
      (($# >= 2)) || die "--task 後面需要 task ID。"
      TASK_ARG=$2
      shift 2
      ;;
    --run)
      (($# >= 2)) || die "--run 後面需要 run 名稱或 latest。"
      RUN_ARG=$2
      shift 2
      ;;
    --checkpoint)
      (($# >= 2)) || die "--checkpoint 後面需要 iteration、檔名或 latest。"
      CHECKPOINT_ARG=$2
      shift 2
      ;;
    --output-name)
      (($# >= 2)) || die "--output-name 後面需要檔名。"
      OUTPUT_NAME_ARG=$2
      shift 2
      ;;
    --device)
      (($# >= 2)) || die "--device 後面需要 device，例如 cpu 或 cuda:0。"
      DEVICE=$2
      shift 2
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --no-metadata)
      NO_METADATA=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --list-tasks)
      LIST_TASKS=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "未知選項：$1"
      ;;
  esac
done

if [[ -z ${MPLCONFIGDIR:-} ]]; then
  export MPLCONFIGDIR="${TMPDIR:-/tmp}/mjlab-matplotlib-${UID}"
fi
if ! mkdir -p -- "$MPLCONFIGDIR" || [[ ! -w $MPLCONFIGDIR ]]; then
  die "Matplotlib 設定目錄無法寫入：$MPLCONFIGDIR"
fi

prompt_text() {
  local label=$1
  local default_value=${2-}
  local allow_empty=${3:-false}

  while true; do
    if [[ -n $default_value ]]; then
      printf '%s [%s]：' "$label" "$default_value"
    elif [[ $allow_empty == true ]]; then
      printf '%s [可留空]：' "$label"
    else
      printf '%s：' "$label"
    fi

    if ! IFS= read -r ANSWER; then
      cancel
    fi
    if [[ -z $ANSWER ]]; then
      ANSWER=$default_value
    fi
    if [[ -n $ANSWER || $allow_empty == true ]]; then
      return
    fi
    printf '請輸入內容。\n' >&2
  done
}

prompt_yes_no() {
  local label=$1
  local default_value=${2:-no}
  local hint='y/N'

  if [[ $default_value == yes ]]; then
    hint='Y/n'
  fi

  while true; do
    printf '%s [%s]：' "$label" "$hint"
    if ! IFS= read -r ANSWER; then
      cancel
    fi
    if [[ -z $ANSWER ]]; then
      ANSWER=$default_value
    fi
    case "$ANSWER" in
      [Yy] | [Yy][Ee][Ss])
        ANSWER=True
        return
        ;;
      [Nn] | [Nn][Oo])
        ANSWER=False
        return
        ;;
      *)
        printf '請輸入 y 或 n。\n' >&2
        ;;
    esac
  done
}

choose_option() {
  local label=$1
  local default_index=$2
  shift 2
  local options=("$@")
  local index

  while true; do
    printf '\n%s\n' "$label"
    for index in "${!options[@]}"; do
      printf '  %d) %s\n' "$((index + 1))" "${options[$index]}"
    done
    printf '  q) 取消\n'
    printf '請選擇 [%s]：' "$default_index"

    if ! IFS= read -r ANSWER; then
      cancel
    fi
    if [[ -z $ANSWER ]]; then
      ANSWER=$default_index
    fi
    case "$ANSWER" in
      [Qq] | [Qq][Uu][Ii][Tt] | [Ee][Xx][Ii][Tt])
        cancel
        ;;
    esac
    if [[ $ANSWER =~ ^[0-9]+$ ]] &&
      ((10#$ANSWER >= 1 && 10#$ANSWER <= ${#options[@]})); then
      ANSWER=$((10#$ANSWER - 1))
      return
    fi
    printf '請輸入 1 到 %d，或輸入 q。\n' "${#options[@]}" >&2
  done
}

choose_paged_option() {
  local label=$1
  local default_index=$2
  local page_size=$3
  shift 3
  local options=("$@")
  local page=0
  local page_count=$(((${#options[@]} + page_size - 1) / page_size))
  local start
  local end
  local index
  local page_default

  while true; do
    start=$((page * page_size))
    end=$((start + page_size))
    ((end > ${#options[@]})) && end=${#options[@]}
    page_default=$((start + 1))
    if ((default_index > start && default_index <= end)); then
      page_default=$default_index
    fi

    printf '\n%s（第 %d/%d 頁）\n' "$label" "$((page + 1))" "$page_count"
    for ((index = start; index < end; index += 1)); do
      printf '  %d) %s\n' "$((index + 1))" "${options[$index]}"
    done
    ((page + 1 < page_count)) && printf '  n) 下一頁\n'
    ((page > 0)) && printf '  p) 上一頁\n'
    printf '  q) 取消\n'
    printf '請選擇 [%s]：' "$page_default"

    if ! IFS= read -r ANSWER; then
      cancel
    fi
    if [[ -z $ANSWER ]]; then
      ANSWER=$page_default
    fi
    case "$ANSWER" in
      [Qq] | [Qq][Uu][Ii][Tt] | [Ee][Xx][Ii][Tt])
        cancel
        ;;
      n | next)
        if ((page + 1 < page_count)); then
          page=$((page + 1))
        else
          printf '已經是最後一頁。\n' >&2
        fi
        continue
        ;;
      p | prev | previous)
        if ((page > 0)); then
          page=$((page - 1))
        else
          printf '已經是第一頁。\n' >&2
        fi
        continue
        ;;
    esac
    if [[ $ANSWER =~ ^[0-9]+$ ]] &&
      ((10#$ANSWER >= 1 && 10#$ANSWER <= ${#options[@]})); then
      ANSWER=$((10#$ANSWER - 1))
      return
    fi
    printf '請輸入 1 到 %d、n、p 或 q。\n' "${#options[@]}" >&2
  done
}

resolve_path() {
  local candidate=$1
  local base_dir=$2
  local candidate_dir
  local candidate_base

  if [[ $candidate == '~' ]]; then
    candidate=${HOME:?}
  elif [[ $candidate == '~/'* ]]; then
    candidate="${HOME:?}/${candidate:2}"
  elif [[ $candidate != /* ]]; then
    candidate="$base_dir/$candidate"
  fi

  candidate_dir="$(dirname -- "$candidate")"
  candidate_base="$(basename -- "$candidate")"
  (
    cd -- "$candidate_dir" 2>/dev/null
    local resolved_dir
    resolved_dir="$(pwd -P)"
    if [[ $resolved_dir == "/" ]]; then
      printf '/%s\n' "$candidate_base"
    else
      printf '%s/%s\n' "$resolved_dir" "$candidate_base"
    fi
  ) || printf '%s\n' "$candidate"
}

resolve_existing_project_dir() {
  local candidate=$1
  local base_dir=$2

  if [[ $candidate == '~' ]]; then
    candidate=${HOME:?}
  elif [[ $candidate == '~/'* ]]; then
    candidate="${HOME:?}/${candidate:2}"
  elif [[ $candidate != /* ]]; then
    candidate="$base_dir/$candidate"
  fi

  [[ -d $candidate ]] || die "找不到專案目錄：$candidate"
  (
    cd -- "$candidate"
    pwd -P
  )
}

select_project() {
  local invocation_dir=$PWD
  local parent_dir
  local sibling_dir
  local selected_index
  local -a candidates=("$CORE_PROJECT_DIR")
  local -a labels=("$(basename -- "$CORE_PROJECT_DIR")")

  if [[ -n $PROJECT_ARG ]]; then
    RUN_PROJECT="$(resolve_existing_project_dir "$PROJECT_ARG" "$invocation_dir")"
    return
  fi

  parent_dir="$(cd -- "$CORE_PROJECT_DIR/.." && pwd -P)"
  for sibling_dir in "$parent_dir/mjlab_playground_test" "$parent_dir/mjlab_playground"; do
    if [[ -f $sibling_dir/pyproject.toml && -x $sibling_dir/.venv/bin/python ]]; then
      candidates+=("$sibling_dir")
      labels+=("$(basename -- "$sibling_dir")")
    fi
  done

  if ((${#candidates[@]} == 1)) || [[ ! -t 0 ]]; then
    RUN_PROJECT=${candidates[0]}
    return
  fi

  choose_option "選擇匯出來源專案：" 1 "${labels[@]}"
  selected_index=$ANSWER
  RUN_PROJECT=${candidates[$selected_index]}
}

append_pythonpath() {
  local path=$1
  [[ -d $path ]] || return
  if [[ -z ${PYTHONPATH:-} ]]; then
    export PYTHONPATH="$path"
  else
    export PYTHONPATH="$path:$PYTHONPATH"
  fi
}

select_project
[[ -f $RUN_PROJECT/pyproject.toml ]] || die "目錄不是可辨識的 Python 專案：$RUN_PROJECT"

PYTHON_BIN="$RUN_PROJECT/.venv/bin/python"
[[ -x $PYTHON_BIN ]] || die "找不到 Python venv：$PYTHON_BIN"

if [[ -z ${WARP_CACHE_PATH:-} ]]; then
  export WARP_CACHE_PATH="$RUN_PROJECT/.cache/warp"
fi
mkdir -p -- "$WARP_CACHE_PATH"

append_pythonpath "$CORE_PROJECT_DIR/src"
append_pythonpath "$RUN_PROJECT/src"

if [[ -n $LOG_ROOT_ARG ]]; then
  LOG_ROOT_PATH="$(resolve_path "$LOG_ROOT_ARG" "$RUN_PROJECT")"
elif [[ ! -t 0 ]]; then
  LOG_ROOT_PATH="$(resolve_path "logs/rsl_rl" "$RUN_PROJECT")"
else
  prompt_text "RSL-RL log 根目錄" "logs/rsl_rl"
  LOG_ROOT_PATH="$(resolve_path "$ANSWER" "$RUN_PROJECT")"
fi
[[ -d $LOG_ROOT_PATH ]] || die "找不到 log 根目錄：$LOG_ROOT_PATH"

printf '\n[INFO] 從 %s 讀取 task 和 checkpoints...\n' "$RUN_PROJECT"
if ! TASK_DATA="$(
  cd -- "$RUN_PROJECT"
  MJLAB_LOG_ROOT="$LOG_ROOT_PATH" "$PYTHON_BIN" - <<'PY'
import importlib
import os
import re
from pathlib import Path

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_rl_cfg

try:
  importlib.import_module("mjlab_playground")
except Exception:
  pass

log_root = Path(os.environ["MJLAB_LOG_ROOT"])
checkpoint_re = re.compile(r"^model_(\d+)\.pt$")


def checkpoint_iter(path: Path) -> int:
  match = checkpoint_re.fullmatch(path.name)
  return int(match.group(1)) if match else -1


def experiment_checkpoints(experiment_name: str) -> list[Path]:
  experiment_dir = log_root / experiment_name
  if not experiment_dir.is_dir():
    return []
  checkpoints = []
  for path in experiment_dir.rglob("model_*.pt"):
    if "wandb_checkpoints" in path.parts:
      continue
    if path.is_file() and checkpoint_re.fullmatch(path.name):
      checkpoints.append(path)
  return checkpoints


records = []
for task_id in list_tasks():
  agent_cfg = load_rl_cfg(task_id)
  experiment_name = agent_cfg.experiment_name
  checkpoints = experiment_checkpoints(experiment_name)
  run_count = len({path.parent for path in checkpoints})
  latest = max(
    checkpoints,
    key=lambda path: (checkpoint_iter(path), path.stat().st_mtime_ns),
    default=None,
  )
  records.append((len(checkpoints) == 0, task_id, experiment_name, run_count, len(checkpoints), latest))

for _, task_id, experiment_name, run_count, checkpoint_count, latest in sorted(records):
  latest_text = "-" if latest is None else str(latest)
  print(
    "\t".join(
      (
        "__MJLAB_TASK__",
        task_id,
        experiment_name,
        str(run_count),
        str(checkpoint_count),
        latest_text,
      )
    )
  )
PY
)"; then
  die "無法讀取 task；請確認 $RUN_PROJECT 的 Python 環境可 import mjlab。"
fi

TASK_IDS=()
EXPERIMENT_NAMES=()
RUN_COUNTS=()
CHECKPOINT_COUNTS=()
LATEST_CHECKPOINTS=()
TASK_LABELS=()

while IFS=$'\t' read -r marker task_id experiment_name run_count checkpoint_count latest_checkpoint; do
  [[ $marker == __MJLAB_TASK__ ]] || continue
  TASK_IDS+=("$task_id")
  EXPERIMENT_NAMES+=("$experiment_name")
  RUN_COUNTS+=("$run_count")
  CHECKPOINT_COUNTS+=("$checkpoint_count")
  LATEST_CHECKPOINTS+=("$latest_checkpoint")
  TASK_LABELS+=("$task_id（experiment=$experiment_name, runs=$run_count, checkpoints=$checkpoint_count）")
done <<<"$TASK_DATA"

((${#TASK_IDS[@]} > 0)) || die "沒有找到已註冊的 mjlab task。"

if [[ $LIST_TASKS == true ]]; then
  printf '\nRegistered tasks:\n'
  for index in "${!TASK_IDS[@]}"; do
    printf '%s\n' "- ${TASK_LABELS[$index]}"
    printf '    latest: %s\n' "${LATEST_CHECKPOINTS[$index]}"
  done
  exit 0
fi

if [[ -n $TASK_ARG ]]; then
  TASK_INDEX=-1
  for index in "${!TASK_IDS[@]}"; do
    if [[ ${TASK_IDS[$index]} == "$TASK_ARG" ]]; then
      TASK_INDEX=$index
      break
    fi
  done
  if ((TASK_INDEX < 0)); then
    die "找不到 task：$TASK_ARG。可用 --list-tasks 查看。"
  fi
else
  choose_paged_option "選擇要輸出的任務：" 1 15 "${TASK_LABELS[@]}"
  TASK_INDEX=$ANSWER
fi

TASK_ID=${TASK_IDS[$TASK_INDEX]}
EXPERIMENT_NAME=${EXPERIMENT_NAMES[$TASK_INDEX]}
EXPERIMENT_DIR="$LOG_ROOT_PATH/$EXPERIMENT_NAME"
[[ -d $EXPERIMENT_DIR ]] || die "找不到實驗紀錄目錄：$EXPERIMENT_DIR"

if ! RUN_DATA="$(
  MJLAB_EXPERIMENT_DIR="$EXPERIMENT_DIR" "$PYTHON_BIN" - <<'PY'
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

experiment_dir = Path(os.environ["MJLAB_EXPERIMENT_DIR"])
checkpoint_re = re.compile(r"^model_(\d+)\.pt$")
runs: dict[Path, list[Path]] = defaultdict(list)

for checkpoint in experiment_dir.rglob("model_*.pt"):
  if "wandb_checkpoints" in checkpoint.parts:
    continue
  if checkpoint.is_file() and checkpoint_re.fullmatch(checkpoint.name):
    runs[checkpoint.parent].append(checkpoint)


def checkpoint_iter(path: Path) -> int:
  return int(checkpoint_re.fullmatch(path.name).group(1))


records = []
for run_dir, checkpoints in runs.items():
  latest = max(
    checkpoints,
    key=lambda path: (checkpoint_iter(path), path.stat().st_mtime_ns),
  )
  records.append((latest.stat().st_mtime_ns, run_dir, checkpoints, latest))

for _, run_dir, checkpoints, latest in sorted(records, reverse=True):
  mtime = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
  print(
    "\t".join(
      (
        "__MJLAB_RUN__",
        run_dir.name,
        str(run_dir),
        str(len(checkpoints)),
        str(checkpoint_iter(latest)),
        mtime,
      )
    )
  )
PY
)"; then
  die "無法讀取 run：$EXPERIMENT_DIR"
fi

RUN_NAMES=()
RUN_DIRS=()
RUN_LABELS=()

while IFS=$'\t' read -r marker run_name run_dir checkpoint_count latest_iter latest_mtime; do
  [[ $marker == __MJLAB_RUN__ ]] || continue
  RUN_NAMES+=("$run_name")
  RUN_DIRS+=("$run_dir")
  RUN_LABELS+=("$run_name（checkpoints=$checkpoint_count, latest=model_${latest_iter}.pt, $latest_mtime）")
done <<<"$RUN_DATA"

((${#RUN_NAMES[@]} > 0)) || die "$EXPERIMENT_DIR 下沒有 model_*.pt。"

if [[ -n $RUN_ARG ]]; then
  if [[ $RUN_ARG == latest ]]; then
    RUN_INDEX=0
  else
    RUN_INDEX=-1
    for index in "${!RUN_NAMES[@]}"; do
      if [[ ${RUN_NAMES[$index]} == "$RUN_ARG" ]]; then
        RUN_INDEX=$index
        break
      fi
    done
    ((RUN_INDEX >= 0)) || die "找不到 run：$RUN_ARG"
  fi
else
  choose_paged_option "選擇 run：" 1 15 "${RUN_LABELS[@]}"
  RUN_INDEX=$ANSWER
fi

RUN_NAME=${RUN_NAMES[$RUN_INDEX]}
RUN_DIR=${RUN_DIRS[$RUN_INDEX]}

if ! CHECKPOINT_DATA="$(
  MJLAB_RUN_DIR="$RUN_DIR" "$PYTHON_BIN" - <<'PY'
import os
import re
from datetime import datetime
from pathlib import Path

run_dir = Path(os.environ["MJLAB_RUN_DIR"])
checkpoint_re = re.compile(r"^model_(\d+)\.pt$")


def checkpoint_iter(path: Path) -> int:
  return int(checkpoint_re.fullmatch(path.name).group(1))


checkpoints = [
  path
  for path in run_dir.glob("model_*.pt")
  if path.is_file() and checkpoint_re.fullmatch(path.name)
]

for path in sorted(checkpoints, key=lambda p: (checkpoint_iter(p), p.stat().st_mtime_ns), reverse=True):
  mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
  print("\t".join(("__MJLAB_CHECKPOINT__", str(checkpoint_iter(path)), path.name, str(path), mtime)))
PY
)"; then
  die "無法讀取 checkpoint：$RUN_DIR"
fi

CHECKPOINT_ITERS=()
CHECKPOINT_NAMES=()
CHECKPOINT_PATHS=()
CHECKPOINT_LABELS=()

while IFS=$'\t' read -r marker checkpoint_iter checkpoint_name checkpoint_path checkpoint_mtime; do
  [[ $marker == __MJLAB_CHECKPOINT__ ]] || continue
  CHECKPOINT_ITERS+=("$checkpoint_iter")
  CHECKPOINT_NAMES+=("$checkpoint_name")
  CHECKPOINT_PATHS+=("$checkpoint_path")
  CHECKPOINT_LABELS+=("$checkpoint_name（iteration=$checkpoint_iter, $checkpoint_mtime）")
done <<<"$CHECKPOINT_DATA"

((${#CHECKPOINT_PATHS[@]} > 0)) || die "$RUN_DIR 下沒有 model_*.pt。"

if [[ -n $CHECKPOINT_ARG ]]; then
  if [[ $CHECKPOINT_ARG == latest ]]; then
    CHECKPOINT_INDEX=0
  else
    CHECKPOINT_INDEX=-1
    WANTED_NAME=$CHECKPOINT_ARG
    if [[ $CHECKPOINT_ARG =~ ^[0-9]+$ ]]; then
      WANTED_NAME="model_${CHECKPOINT_ARG}.pt"
    fi
    for index in "${!CHECKPOINT_NAMES[@]}"; do
      if [[ ${CHECKPOINT_NAMES[$index]} == "$WANTED_NAME" || ${CHECKPOINT_PATHS[$index]} == "$CHECKPOINT_ARG" ]]; then
        CHECKPOINT_INDEX=$index
        break
      fi
    done
    ((CHECKPOINT_INDEX >= 0)) || die "在 $RUN_NAME 找不到 checkpoint：$CHECKPOINT_ARG"
  fi
else
  choose_paged_option "選擇要輸出的 checkpoint 步數：" 1 20 "${CHECKPOINT_LABELS[@]}"
  CHECKPOINT_INDEX=$ANSWER
fi

CHECKPOINT_ITER=${CHECKPOINT_ITERS[$CHECKPOINT_INDEX]}
CHECKPOINT_NAME=${CHECKPOINT_NAMES[$CHECKPOINT_INDEX]}
CHECKPOINT_PATH=${CHECKPOINT_PATHS[$CHECKPOINT_INDEX]}

if [[ -n $OUTPUT_ROOT_ARG ]]; then
  OUTPUT_ROOT="$(resolve_path "$OUTPUT_ROOT_ARG" "$RUN_PROJECT")"
elif [[ ! -t 0 ]]; then
  OUTPUT_ROOT="$RUN_PROJECT/policy_export"
else
  OUTPUT_ROOT_OPTIONS=("$RUN_PROJECT/policy_export")
  OUTPUT_ROOT_LABELS=("$(basename -- "$RUN_PROJECT")/policy_export")
  QUAD_PPO_DIR="$(cd -- "$CORE_PROJECT_DIR/.." && pwd -P)/quadruped_robot_simulation/policy/ppo"
  if [[ -d $QUAD_PPO_DIR ]]; then
    OUTPUT_ROOT_OPTIONS+=("$QUAD_PPO_DIR")
    OUTPUT_ROOT_LABELS+=("quadruped_robot_simulation/policy/ppo")
  fi
  OUTPUT_ROOT_OPTIONS+=("__custom__")
  OUTPUT_ROOT_LABELS+=("自訂輸出目錄")
  choose_option "選擇 ONNX 輸出位置：" 1 "${OUTPUT_ROOT_LABELS[@]}"
  OUTPUT_ROOT=${OUTPUT_ROOT_OPTIONS[$ANSWER]}
  if [[ $OUTPUT_ROOT == "__custom__" ]]; then
    prompt_text "ONNX 輸出目錄" "policy_export"
    OUTPUT_ROOT="$(resolve_path "$ANSWER" "$RUN_PROJECT")"
  fi
fi
mkdir -p -- "$OUTPUT_ROOT"

DEFAULT_OUTPUT_NAME="${EXPERIMENT_NAME}_${RUN_NAME}_model_${CHECKPOINT_ITER}.onnx"
if [[ -n $OUTPUT_NAME_ARG ]]; then
  OUTPUT_NAME=$OUTPUT_NAME_ARG
elif [[ ! -t 0 ]]; then
  OUTPUT_NAME=$DEFAULT_OUTPUT_NAME
else
  prompt_text "ONNX 輸出檔名" "$DEFAULT_OUTPUT_NAME"
  OUTPUT_NAME=$ANSWER
fi
[[ $OUTPUT_NAME == *.onnx ]] || OUTPUT_NAME="${OUTPUT_NAME}.onnx"
OUTPUT_PATH="$OUTPUT_ROOT/$OUTPUT_NAME"

if [[ -e $OUTPUT_PATH && $FORCE != true ]]; then
  if [[ ! -t 0 ]]; then
    die "輸出已存在：$OUTPUT_PATH"
  fi
  prompt_yes_no "輸出已存在，是否覆寫？$OUTPUT_PATH" no
  [[ $ANSWER == True ]] || die "輸出已存在：$OUTPUT_PATH"
  FORCE=true
fi

printf '\nExport 設定\n'
printf '  Project   : %s\n' "$RUN_PROJECT"
printf '  Task      : %s\n' "$TASK_ID"
printf '  Experiment: %s\n' "$EXPERIMENT_NAME"
printf '  Run       : %s\n' "$RUN_NAME"
printf '  Checkpoint: %s\n' "$CHECKPOINT_PATH"
printf '  Output    : %s\n' "$OUTPUT_PATH"
printf '  Device    : %s\n' "$DEVICE"
printf '  Metadata  : %s\n' "$([[ $NO_METADATA == true ]] && printf 'no' || printf 'yes')"

if [[ $DRY_RUN == true ]]; then
  printf '\n[DRY-RUN] 未執行匯出。\n'
  exit 0
fi

if [[ -t 0 ]] && [[ -z $TASK_ARG || -z $RUN_ARG || -z $CHECKPOINT_ARG || -z $OUTPUT_NAME_ARG ]]; then
  prompt_yes_no "開始匯出？" yes
  [[ $ANSWER == True ]] || cancel
fi

printf '\n[INFO] 開始匯出 ONNX...\n'
(
  cd -- "$RUN_PROJECT"
  MJLAB_TASK_ID="$TASK_ID" \
    MJLAB_RUN_DIR="$RUN_DIR" \
    MJLAB_CHECKPOINT_PATH="$CHECKPOINT_PATH" \
    MJLAB_OUTPUT_PATH="$OUTPUT_PATH" \
    MJLAB_DEVICE="$DEVICE" \
    MJLAB_INCLUDE_METADATA="$([[ $NO_METADATA == true ]] && printf '0' || printf '1')" \
    MJLAB_FORCE="$([[ $FORCE == true ]] && printf '1' || printf '0')" \
    MJLAB_VERBOSE="$([[ $VERBOSE == true ]] && printf '1' || printf '0')" \
    "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import importlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mjlab.tasks  # noqa: F401
import onnx
import torch
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

try:
  importlib.import_module("mjlab_playground")
except Exception:
  pass


class SafeAgentConfigLoader(yaml.SafeLoader):
  """Safe YAML loader with tuple support for configs written by mjlab."""


def construct_python_tuple(
  loader: SafeAgentConfigLoader, node: yaml.nodes.SequenceNode
) -> tuple[Any, ...]:
  return tuple(loader.construct_sequence(node, deep=True))


SafeAgentConfigLoader.add_constructor(
  "tag:yaml.org,2002:python/tuple", construct_python_tuple
)


def load_agent_cfg(run_dir: Path, task_id: str) -> dict[str, Any]:
  config_path = run_dir / "params" / "agent.yaml"
  if config_path.is_file():
    loaded = yaml.load(config_path.read_text(), Loader=SafeAgentConfigLoader)
    if not isinstance(loaded, dict):
      raise RuntimeError(f"Agent config is not a mapping: {config_path}")
    return loaded
  print("[WARN] 找不到 run/params/agent.yaml，改用目前 task registry 設定。")
  return asdict(load_rl_cfg(task_id))


def shape_text(value_info: onnx.ValueInfoProto) -> str:
  dims = []
  for dim in value_info.type.tensor_type.shape.dim:
    dims.append(str(dim.dim_value or dim.dim_param or "?"))
  return f"{value_info.name}[{', '.join(dims)}]"


task_id = os.environ["MJLAB_TASK_ID"]
run_dir = Path(os.environ["MJLAB_RUN_DIR"])
checkpoint_path = Path(os.environ["MJLAB_CHECKPOINT_PATH"])
output_path = Path(os.environ["MJLAB_OUTPUT_PATH"])
device = os.environ["MJLAB_DEVICE"]
include_metadata = os.environ["MJLAB_INCLUDE_METADATA"] == "1"
force = os.environ["MJLAB_FORCE"] == "1"
verbose = os.environ["MJLAB_VERBOSE"] == "1"

if output_path.exists() and not force:
  raise RuntimeError(f"Output already exists: {output_path}")

env = None
with tempfile.NamedTemporaryFile(
  prefix=f".{output_path.stem}.",
  suffix=".tmp.onnx",
  dir=output_path.parent,
  delete=False,
) as temporary_file:
  temporary_path = Path(temporary_file.name)

try:
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_agent_cfg(run_dir, task_id)
  clip_actions = agent_cfg.get("clip_actions")

  print(f"[INFO] 建立環境：{task_id} ({device})")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, agent_cfg, log_dir=None, device=device)

  print(f"[INFO] 載入 checkpoint：{checkpoint_path}")
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )

  print("[INFO] 匯出 ONNX...")
  runner.export_policy_to_onnx(
    str(temporary_path.parent), temporary_path.name, verbose=verbose
  )

  if include_metadata:
    metadata = get_base_metadata(env, run_dir.name)
    metadata["task_id"] = task_id
    metadata["experiment_run_dir"] = str(run_dir)
    metadata["checkpoint_name"] = checkpoint_path.name
    metadata["checkpoint_path"] = str(checkpoint_path)
    attach_metadata_to_onnx(str(temporary_path), metadata)

  model = onnx.load(str(temporary_path))
  onnx.checker.check_model(model)
  input_shapes = [shape_text(item) for item in model.graph.input]
  output_shapes = [shape_text(item) for item in model.graph.output]

  temporary_path.chmod(0o644)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  if output_path.exists() and force:
    output_path.unlink()
  temporary_path.replace(output_path)

  print("[OK] ONNX 匯出完成")
  print(f"  Output : {output_path}")
  print(f"  Inputs : {', '.join(input_shapes)}")
  print(f"  Outputs: {', '.join(output_shapes)}")
finally:
  temporary_path.unlink(missing_ok=True)
  if env is not None:
    env.close()
PY
)
