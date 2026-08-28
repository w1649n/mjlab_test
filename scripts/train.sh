#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CORE_PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

DRY_RUN=false
PROJECT_ARG=""
EXTRA_ARGS=()
ANSWER=""

usage() {
  cat <<'EOF'
互動式啟動 mjlab 訓練。

用法：
  ./scripts/train.sh [--project DIR] [--dry-run] [-- 額外 train 參數...]

選項：
  --project DIR  指定要執行的 mjlab 專案（例如 ../mjlab_playground）
  --dry-run      產生並顯示最後命令，不開始訓練
  -h, --help     顯示此說明

範例：
  ./scripts/train.sh
  ./scripts/train.sh --project ../mjlab_playground
  ./scripts/train.sh --dry-run -- --env.episode-length-s 10

「--」後的參數會安全地附加到互動選項之後，可用來覆寫進階設定。
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
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      die "未知選項：$1（進階 train 參數請放在 -- 後面）"
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

prompt_uint() {
  local label=$1
  local default_value=$2
  local minimum=${3:-0}

  while true; do
    prompt_text "$label" "$default_value"
    if [[ $ANSWER =~ ^[0-9]+$ ]] && ((10#$ANSWER >= minimum)); then
      return
    fi
    printf '請輸入不小於 %s 的整數。\n' "$minimum" >&2
  done
}

prompt_positive_number() {
  local label=$1
  local default_value=$2
  local number_pattern='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'

  while true; do
    prompt_text "$label" "$default_value"
    if [[ $ANSWER =~ $number_pattern ]] &&
      [[ ! $ANSWER =~ ^(0+([.]0*)?|[.]0+)([eE][+-]?[0-9]+)?$ ]]; then
      return
    fi
    printf '請輸入大於 0 的數字。\n' >&2
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

resolve_project_dir() {
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
  local sibling_dir
  local selected_index
  local -a candidates=("$CORE_PROJECT_DIR")
  local -a labels=("$(basename -- "$CORE_PROJECT_DIR")（核心任務）")

  if [[ -n $PROJECT_ARG ]]; then
    RUN_PROJECT="$(resolve_project_dir "$PROJECT_ARG" "$invocation_dir")"
    return
  fi

  sibling_dir="$(cd -- "$CORE_PROJECT_DIR/.." && pwd -P)/mjlab_playground"
  if [[ -f $sibling_dir/pyproject.toml &&
    -x $sibling_dir/.venv/bin/python ]]; then
    candidates+=("$sibling_dir")
    labels+=("$(basename -- "$sibling_dir")（包含外掛／Getup 任務）")
  fi

  if ((${#candidates[@]} == 1)); then
    RUN_PROJECT=${candidates[0]}
    return
  fi

  choose_option "選擇執行環境：" 1 "${labels[@]}"
  selected_index=$ANSWER
  RUN_PROJECT=${candidates[$selected_index]}
}

select_project
[[ -f $RUN_PROJECT/pyproject.toml ]] ||
  die "目錄不是可辨識的 Python 專案：$RUN_PROJECT"
command -v uv >/dev/null 2>&1 || die "找不到 uv，請先安裝 uv。"

UV_COMMAND=(uv run)
if [[ -x $RUN_PROJECT/.venv/bin/python ]]; then
  UV_COMMAND+=(--no-sync)
fi

printf '\n[INFO] 從 %s 讀取已註冊任務...\n' "$RUN_PROJECT"
if ! TASK_DATA="$({
  cd -- "$RUN_PROJECT"
  "${UV_COMMAND[@]}" python - <<'PY'
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

for task_id in list_tasks():
  env = load_env_cfg(task_id)
  agent = load_rl_cfg(task_id)
  motion = getattr(env, "commands", {}).get("motion")
  values = (
    "__MJLAB_TASK__",
    task_id,
    env.scene.num_envs,
    agent.max_iterations,
    agent.num_steps_per_env,
    agent.seed,
    agent.save_interval,
    agent.logger,
    agent.experiment_name,
    agent.wandb_project,
    agent.upload_model,
    agent.algorithm.learning_rate,
    isinstance(motion, MotionCommandCfg),
  )
  print("\t".join(str(value) for value in values))
PY
})"; then
  die "無法讀取任務；請確認所選專案已完成 uv sync。"
fi

TASK_IDS=()
DEFAULT_NUM_ENVS=()
DEFAULT_ITERATIONS=()
DEFAULT_STEPS=()
DEFAULT_SEEDS=()
DEFAULT_SAVE_INTERVALS=()
DEFAULT_LOGGERS=()
EXPERIMENT_NAMES=()
DEFAULT_WANDB_PROJECTS=()
DEFAULT_UPLOAD_MODELS=()
DEFAULT_LEARNING_RATES=()
IS_TRACKING=()

while IFS=$'\t' read -r marker task_id num_envs iterations steps seed \
  save_interval logger experiment_name wandb_project upload_model learning_rate \
  is_tracking; do
  [[ $marker == __MJLAB_TASK__ ]] || continue
  [[ -n $task_id ]] || continue
  TASK_IDS+=("$task_id")
  DEFAULT_NUM_ENVS+=("$num_envs")
  DEFAULT_ITERATIONS+=("$iterations")
  DEFAULT_STEPS+=("$steps")
  DEFAULT_SEEDS+=("$seed")
  DEFAULT_SAVE_INTERVALS+=("$save_interval")
  DEFAULT_LOGGERS+=("$logger")
  EXPERIMENT_NAMES+=("$experiment_name")
  DEFAULT_WANDB_PROJECTS+=("$wandb_project")
  DEFAULT_UPLOAD_MODELS+=("$upload_model")
  DEFAULT_LEARNING_RATES+=("$learning_rate")
  IS_TRACKING+=("$is_tracking")
done <<<"$TASK_DATA"

((${#TASK_IDS[@]} > 0)) || die "所選環境沒有註冊任何 mjlab 任務。"

choose_option "選擇訓練任務：" 1 "${TASK_IDS[@]}"
TASK_INDEX=$ANSWER
TASK_ID=${TASK_IDS[$TASK_INDEX]}

printf '\n已選擇：%s\n' "$TASK_ID"
printf 'Experiment：%s\n\n' "${EXPERIMENT_NAMES[$TASK_INDEX]}"

prompt_uint \
  "平行環境數（任務預設 ${DEFAULT_NUM_ENVS[$TASK_INDEX]}）" \
  "${DEFAULT_NUM_ENVS[$TASK_INDEX]}" 1
NUM_ENVS=$ANSWER

prompt_uint "最大訓練 iteration" \
  "${DEFAULT_ITERATIONS[$TASK_INDEX]}" 1
MAX_ITERATIONS=$ANSWER

prompt_uint "每個環境每次 rollout 的步數" \
  "${DEFAULT_STEPS[$TASK_INDEX]}" 1
NUM_STEPS=$ANSWER

prompt_positive_number "Learning rate" \
  "${DEFAULT_LEARNING_RATES[$TASK_INDEX]}"
LEARNING_RATE=$ANSWER

prompt_uint "隨機種子" "${DEFAULT_SEEDS[$TASK_INDEX]}" 0
SEED=$ANSWER

prompt_uint "Checkpoint 儲存間隔" \
  "${DEFAULT_SAVE_INTERVALS[$TASK_INDEX]}" 1
SAVE_INTERVAL=$ANSWER

prompt_text "Run 名稱" "" true
RUN_NAME=$ANSWER

prompt_text "Log 根目錄（相對於所選專案）" "logs/rsl_rl"
LOG_ROOT=$ANSWER

LOGGER_OPTIONS=(wandb tensorboard)
LOGGER_DEFAULT=1
if [[ ${DEFAULT_LOGGERS[$TASK_INDEX]} == tensorboard ]]; then
  LOGGER_DEFAULT=2
fi
choose_option "選擇 logger：" "$LOGGER_DEFAULT" "${LOGGER_OPTIONS[@]}"
LOGGER=${LOGGER_OPTIONS[$ANSWER]}

WANDB_PROJECT=""
UPLOAD_MODEL=""
if [[ $LOGGER == wandb ]]; then
  prompt_text "W&B project" "${DEFAULT_WANDB_PROJECTS[$TASK_INDEX]}"
  WANDB_PROJECT=$ANSWER
  upload_default=no
  if [[ ${DEFAULT_UPLOAD_MODELS[$TASK_INDEX]} == True ]]; then
    upload_default=yes
  fi
  prompt_yes_no "將模型上傳到 W&B？" "$upload_default"
  UPLOAD_MODEL=$ANSWER
fi

GPU_LABELS=(
  "GPU 0（單 GPU）"
  "所有可見 GPU（多 GPU）"
  "CPU"
  "自訂 GPU 編號"
)
choose_option "選擇訓練裝置：" 1 "${GPU_LABELS[@]}"
case "$ANSWER" in
  0)
    GPU_IDS='[0]'
    ;;
  1)
    GPU_IDS=all
    ;;
  2)
    GPU_IDS=None
    ;;
  3)
    while true; do
      prompt_text "GPU 編號（例如 0 或 0,1）" "0"
      GPU_INPUT=${ANSWER//[[:space:]]/}
      if [[ $GPU_INPUT =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        GPU_IDS="[$GPU_INPUT]"
        break
      fi
      printf '格式錯誤，請輸入逗號分隔的非負整數。\n' >&2
    done
    ;;
esac

prompt_yes_no "訓練期間錄影？" no
VIDEO=$ANSWER
VIDEO_LENGTH=""
VIDEO_INTERVAL=""
if [[ $VIDEO == True ]]; then
  prompt_uint "每段影片長度（步）" 200 1
  VIDEO_LENGTH=$ANSWER
  prompt_uint "錄影間隔（步）" 2000 1
  VIDEO_INTERVAL=$ANSWER
fi

prompt_yes_no "啟用 NaN guard？" no
NAN_GUARD=$ANSWER

MOTION_ARGS=()
if [[ ${IS_TRACKING[$TASK_INDEX]} == True ]]; then
  choose_option "Tracking 任務的 motion 來源：" 1 \
    "本機 motion.npz" "W&B registry artifact"
  if [[ $ANSWER == 0 ]]; then
    while true; do
      prompt_text "motion.npz 路徑（相對於所選專案）" ""
      MOTION_FILE=$ANSWER
      if [[ $MOTION_FILE == '~/'* ]]; then
        MOTION_FILE="${HOME:?}/${MOTION_FILE:2}"
      elif [[ $MOTION_FILE != /* ]]; then
        MOTION_FILE="$RUN_PROJECT/$MOTION_FILE"
      fi
      if [[ -f $MOTION_FILE ]]; then
        MOTION_ARGS=(--env.commands.motion.motion-file "$MOTION_FILE")
        break
      fi
      printf '找不到檔案：%s\n' "$MOTION_FILE" >&2
    done
  else
    prompt_text "W&B registry 名稱（org/project/artifact）" ""
    MOTION_ARGS=(--registry-name "$ANSWER")
  fi
fi

RESUME_ARGS=()
prompt_yes_no "接續既有訓練？" no
if [[ $ANSWER == True ]]; then
  choose_option "Checkpoint 來源：" 1 "最新符合的本機 checkpoint" "W&B run"
  if [[ $ANSWER == 0 ]]; then
    prompt_text "Run 目錄名稱或 regex" '.*'
    LOAD_RUN=$ANSWER
    prompt_text "Checkpoint 檔名或 regex" 'model_.*.pt'
    LOAD_CHECKPOINT=$ANSWER
    RESUME_ARGS=(
      --agent.resume True
      --agent.load-run "$LOAD_RUN"
      --agent.load-checkpoint "$LOAD_CHECKPOINT"
    )
  else
    prompt_text "W&B run path（entity/project/run-id）" ""
    WANDB_RUN_PATH=$ANSWER
    prompt_text "W&B checkpoint 名稱" "" true
    WANDB_CHECKPOINT=$ANSWER
    RESUME_ARGS=(--agent.resume True --wandb-run-path "$WANDB_RUN_PATH")
    if [[ -n $WANDB_CHECKPOINT ]]; then
      RESUME_ARGS+=(--wandb-checkpoint-name "$WANDB_CHECKPOINT")
    fi
  fi
fi

COMMAND=(
  "${UV_COMMAND[@]}" train "$TASK_ID"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max-iterations "$MAX_ITERATIONS"
  --agent.num-steps-per-env "$NUM_STEPS"
  --agent.algorithm.learning-rate "$LEARNING_RATE"
  --agent.seed "$SEED"
  --agent.save-interval "$SAVE_INTERVAL"
  --agent.run-name "$RUN_NAME"
  --agent.logger "$LOGGER"
  --gpu-ids "$GPU_IDS"
  --video "$VIDEO"
  --enable-nan-guard "$NAN_GUARD"
  --log-root "$LOG_ROOT"
)

if [[ $LOGGER == wandb ]]; then
  COMMAND+=(
    --agent.wandb-project "$WANDB_PROJECT"
    --agent.upload-model "$UPLOAD_MODEL"
  )
fi
if [[ $VIDEO == True ]]; then
  COMMAND+=(--video-length "$VIDEO_LENGTH" --video-interval "$VIDEO_INTERVAL")
fi
COMMAND+=("${MOTION_ARGS[@]}" "${RESUME_ARGS[@]}" "${EXTRA_ARGS[@]}")

printf '\n即將執行：\n  cd %q\n  ' "$RUN_PROJECT"
printf '%q ' "${COMMAND[@]}"
printf '\n'

if [[ $DRY_RUN == true ]]; then
  printf '\n[INFO] dry-run 完成，未開始訓練。\n'
  exit 0
fi

prompt_yes_no "開始訓練？" no
[[ $ANSWER == True ]] || cancel

cd -- "$RUN_PROJECT"
exec "${COMMAND[@]}"
