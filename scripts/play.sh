#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CORE_PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

DRY_RUN=false
PROJECT_ARG=""
EXTRA_ARGS=()
ANSWER=""
SORTED_PATHS=()

usage() {
  cat <<'EOF'
互動式啟動 mjlab 策略播放。

用法：
  ./scripts/play.sh [--project DIR] [--dry-run] [-- 額外 play 參數...]

選項：
  --project DIR  指定要執行的 mjlab 專案（例如 ../mjlab_playground）
  --dry-run      產生並顯示最後命令，不開始播放
  -h, --help     顯示此說明

範例：
  ./scripts/play.sh
  ./scripts/play.sh --project ../mjlab_playground
  ./scripts/play.sh --dry-run -- --video-width 1280

「--」後的參數會安全地附加到互動選項之後，可用來覆寫進階設定。
EOF
}

die() {
  printf '[錯誤] %s\n' "$*" >&2
  exit 1
}

cancel() {
  printf '\n[資訊] 已取消。\n'
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
      die "未知選項：$1（進階 play 參數請放在 -- 後面）"
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

prompt_optional_uint() {
  local label=$1
  local minimum=${2:-1}

  while true; do
    prompt_text "$label" "" true
    if [[ -z $ANSWER ]]; then
      return
    fi
    if [[ $ANSWER =~ ^[0-9]+$ ]] && ((10#$ANSWER >= minimum)); then
      return
    fi
    printf '請輸入不小於 %s 的整數，或留空。\n' "$minimum" >&2
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

resolve_file_path() {
  local candidate=$1

  if [[ $candidate == '~' ]]; then
    candidate=${HOME:?}
  elif [[ $candidate == '~/'* ]]; then
    candidate="${HOME:?}/${candidate:2}"
  elif [[ $candidate != /* ]]; then
    candidate="$RUN_PROJECT/$candidate"
  fi
  RESOLVED_FILE=$candidate
}

prompt_existing_file() {
  local label=$1

  while true; do
    prompt_text "$label" ""
    resolve_file_path "$ANSWER"
    if [[ -f $RESOLVED_FILE ]]; then
      return
    fi
    printf '找不到檔案：%s\n' "$RESOLVED_FILE" >&2
  done
}

directory_has_checkpoints() {
  local directory=$1
  local checkpoint

  for checkpoint in "$directory"/*.pt; do
    [[ -f $checkpoint ]] && return 0
  done
  return 1
}

sort_newest_first() {
  local -a paths=("$@")
  local newest_index
  local index
  local -a ordered=()

  while ((${#paths[@]} > 0)); do
    newest_index=0
    for index in "${!paths[@]}"; do
      if [[ ${paths[$index]} -nt ${paths[$newest_index]} ]]; then
        newest_index=$index
      fi
    done
    ordered+=("${paths[$newest_index]}")
    unset 'paths[newest_index]'
    paths=("${paths[@]}")
  done
  SORTED_PATHS=("${ordered[@]}")
}

select_local_checkpoint() {
  local experiment_dir=$1
  local path
  local count
  local selected_index
  local -a run_paths=()
  local -a run_labels=()
  local -a checkpoint_paths=()
  local -a checkpoint_labels=()
  local -a child_paths=()

  [[ -d $experiment_dir ]] ||
    die "找不到實驗紀錄目錄：$experiment_dir"

  if directory_has_checkpoints "$experiment_dir"; then
    run_paths+=("$experiment_dir")
  fi

  child_paths=("$experiment_dir"/*)
  for path in "${child_paths[@]}"; do
    if [[ -d $path ]] && directory_has_checkpoints "$path"; then
      run_paths+=("$path")
    fi
  done
  ((${#run_paths[@]} > 0)) ||
    die "在實驗紀錄目錄下找不到本機模型檢查點：$experiment_dir"

  sort_newest_first "${run_paths[@]}"
  run_paths=("${SORTED_PATHS[@]}")
  for path in "${run_paths[@]}"; do
    count=0
    for checkpoint in "$path"/*.pt; do
      [[ -f $checkpoint ]] && ((count += 1))
    done
    if [[ $path == "$experiment_dir" ]]; then
      run_labels+=("實驗根目錄（${count} 個模型檢查點）")
    else
      run_labels+=("$(basename -- "$path")（${count} 個模型檢查點）")
    fi
  done

  choose_paged_option "選擇本機執行紀錄" 1 15 "${run_labels[@]}"
  selected_index=$ANSWER
  SELECTED_RUN_DIR=${run_paths[$selected_index]}

  for path in "$SELECTED_RUN_DIR"/*.pt; do
    [[ -f $path ]] && checkpoint_paths+=("$path")
  done
  sort_newest_first "${checkpoint_paths[@]}"
  checkpoint_paths=("${SORTED_PATHS[@]}")
  for path in "${checkpoint_paths[@]}"; do
    checkpoint_labels+=("$(basename -- "$path")")
  done

  choose_paged_option "選擇模型檢查點（最新修改的檔案排第一）" 1 20 \
    "${checkpoint_labels[@]}"
  selected_index=$ANSWER
  SELECTED_CHECKPOINT=${checkpoint_paths[$selected_index]}
}

select_project
[[ -f $RUN_PROJECT/pyproject.toml ]] ||
  die "目錄不是可辨識的 Python 專案：$RUN_PROJECT"
command -v uv >/dev/null 2>&1 || die "找不到 uv，請先安裝 uv。"

UV_COMMAND=(uv run)
if [[ -x $RUN_PROJECT/.venv/bin/python ]]; then
  UV_COMMAND+=(--no-sync)
fi

printf '\n[資訊] 從 %s 讀取已註冊任務...\n' "$RUN_PROJECT"
if ! TASK_DATA="$({
  cd -- "$RUN_PROJECT"
  "${UV_COMMAND[@]}" python - <<'PY'
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

for task_id in list_tasks():
  env = load_env_cfg(task_id, play=True)
  agent = load_rl_cfg(task_id)
  motion = getattr(env, "commands", {}).get("motion")
  values = (
    "__MJLAB_TASK__",
    task_id,
    agent.experiment_name,
    env.scene.num_envs,
    isinstance(motion, MotionCommandCfg),
  )
  print("\t".join(str(value) for value in values))
PY
})"; then
  die "無法讀取任務；請確認所選專案已完成 uv sync。"
fi

TASK_IDS=()
EXPERIMENT_NAMES=()
DEFAULT_NUM_ENVS=()
IS_TRACKING=()

while IFS=$'\t' read -r marker task_id experiment_name num_envs is_tracking; do
  [[ $marker == __MJLAB_TASK__ ]] || continue
  [[ -n $task_id ]] || continue
  TASK_IDS+=("$task_id")
  EXPERIMENT_NAMES+=("$experiment_name")
  DEFAULT_NUM_ENVS+=("$num_envs")
  IS_TRACKING+=("$is_tracking")
done <<<"$TASK_DATA"

((${#TASK_IDS[@]} > 0)) || die "所選環境沒有註冊任何 mjlab 任務。"

choose_option "選擇播放任務：" 1 "${TASK_IDS[@]}"
TASK_INDEX=$ANSWER
TASK_ID=${TASK_IDS[$TASK_INDEX]}
EXPERIMENT_NAME=${EXPERIMENT_NAMES[$TASK_INDEX]}

printf '\n已選擇：%s\n' "$TASK_ID"
printf '實驗名稱：%s\n\n' "$EXPERIMENT_NAME"

AGENT_LABELS=(
  "已訓練策略（載入模型檢查點）"
  "零動作策略"
  "隨機動作策略"
)
AGENT_VALUES=(trained zero random)
choose_option "選擇策略：" 1 "${AGENT_LABELS[@]}"
AGENT=${AGENT_VALUES[$ANSWER]}

prompt_uint \
  "平行環境數（任務預設 ${DEFAULT_NUM_ENVS[$TASK_INDEX]}）" \
  "${DEFAULT_NUM_ENVS[$TASK_INDEX]}" 1
NUM_ENVS=$ANSWER

prompt_text "運算裝置（例如 cuda:0 或 cpu；留空則自動選擇）" "" true
DEVICE=$ANSWER

VIEWER_LABELS=("自動選擇" "MuJoCo 原生視窗" "Viser 網頁檢視器")
VIEWER_VALUES=(auto native viser)
choose_option "選擇檢視器：" 1 "${VIEWER_LABELS[@]}"
VIEWER=${VIEWER_VALUES[$ANSWER]}

prompt_yes_no "停用所有終止條件？" no
NO_TERMINATIONS=$ANSWER

SOURCE_ARGS=()
MOTION_ARGS=()
VIDEO_ARGS=()
LOG_ROOT=""
CHECKPOINT_SOURCE=none

if [[ $AGENT == trained ]]; then
  prompt_text "紀錄根目錄（相對於所選專案）" "logs/rsl_rl"
  LOG_ROOT=$ANSWER
  if [[ $LOG_ROOT == '~' ]]; then
    LOG_ROOT=${HOME:?}
    LOG_ROOT_PATH=$LOG_ROOT
  elif [[ $LOG_ROOT == '~/'* ]]; then
    LOG_ROOT="${HOME:?}/${LOG_ROOT:2}"
    LOG_ROOT_PATH=$LOG_ROOT
  elif [[ $LOG_ROOT == /* ]]; then
    LOG_ROOT_PATH=$LOG_ROOT
  else
    LOG_ROOT_PATH="$RUN_PROJECT/$LOG_ROOT"
  fi

  CHECKPOINT_LABELS=(
    "從紀錄目錄自動選擇本機模型檢查點"
    "手動輸入本機模型檢查點路徑"
    "從 W&B 執行紀錄下載"
  )
  choose_option "選擇已訓練策略來源：" 1 "${CHECKPOINT_LABELS[@]}"
  case "$ANSWER" in
    0)
      select_local_checkpoint "$LOG_ROOT_PATH/$EXPERIMENT_NAME"
      SOURCE_ARGS=(--checkpoint-file "$SELECTED_CHECKPOINT")
      CHECKPOINT_SOURCE=local
      ;;
    1)
      prompt_existing_file \
        "模型檢查點路徑（相對於所選專案，副檔名通常為 .pt）"
      SOURCE_ARGS=(--checkpoint-file "$RESOLVED_FILE")
      CHECKPOINT_SOURCE=local
      ;;
    2)
      prompt_text "W&B 執行路徑（entity/project/run-id）" ""
      WANDB_RUN_PATH=$ANSWER
      prompt_text "指定 W&B 模型檢查點名稱" "" true
      WANDB_CHECKPOINT_NAME=$ANSWER
      SOURCE_ARGS=(--wandb-run-path "$WANDB_RUN_PATH")
      if [[ -n $WANDB_CHECKPOINT_NAME ]]; then
        SOURCE_ARGS+=(--wandb-checkpoint-name "$WANDB_CHECKPOINT_NAME")
      fi
      CHECKPOINT_SOURCE=wandb
      ;;
  esac
fi

if [[ ${IS_TRACKING[$TASK_INDEX]} == True ]]; then
  if [[ $AGENT == zero || $AGENT == random ]]; then
    choose_option "Tracking 任務的動作資料來源：" 1 \
      "本機 motion.npz" "W&B registry artifact"
    if [[ $ANSWER == 0 ]]; then
      prompt_existing_file "motion.npz 路徑（相對於所選專案）"
      MOTION_ARGS=(--motion-file "$RESOLVED_FILE")
    else
      prompt_text "W&B registry 名稱（org/project/artifact）" ""
      MOTION_ARGS=(--registry-name "$ANSWER")
    fi
  elif [[ $CHECKPOINT_SOURCE == local ]]; then
    printf '\n本機模型檢查點的 Tracking 任務必須提供動作資料。\n'
    prompt_existing_file "motion.npz 路徑（相對於所選專案）"
    MOTION_ARGS=(--motion-file "$RESOLVED_FILE")
  else
    prompt_yes_no "用本機 motion.npz 覆寫 W&B 執行紀錄的動作資料？" no
    if [[ $ANSWER == True ]]; then
      prompt_existing_file "motion.npz 路徑（相對於所選專案）"
      MOTION_ARGS=(--motion-file "$RESOLVED_FILE")
    else
      printf '[資訊] 將自動使用 W&B 執行紀錄關聯的動作資料。\n'
    fi
  fi
fi

if [[ $AGENT == trained ]]; then
  prompt_yes_no "錄製播放影片？" no
  VIDEO=$ANSWER
  VIDEO_ARGS=(--video "$VIDEO")
  if [[ $VIDEO == True ]]; then
    prompt_uint "影片長度（步）" 200 1
    VIDEO_LENGTH=$ANSWER
    prompt_optional_uint "影片高度（像素）"
    VIDEO_HEIGHT=$ANSWER
    prompt_optional_uint "影片寬度（像素）"
    VIDEO_WIDTH=$ANSWER
    VIDEO_ARGS+=(--video-length "$VIDEO_LENGTH")
    if [[ -n $VIDEO_HEIGHT ]]; then
      VIDEO_ARGS+=(--video-height "$VIDEO_HEIGHT")
    fi
    if [[ -n $VIDEO_WIDTH ]]; then
      VIDEO_ARGS+=(--video-width "$VIDEO_WIDTH")
    fi
  fi
fi

COMMAND=(
  "${UV_COMMAND[@]}" play "$TASK_ID"
  --agent "$AGENT"
  --num-envs "$NUM_ENVS"
  --viewer "$VIEWER"
  --no-terminations "$NO_TERMINATIONS"
)

if [[ -n $DEVICE ]]; then
  COMMAND+=(--device "$DEVICE")
fi
if [[ $AGENT == trained ]]; then
  COMMAND+=(--log-root "$LOG_ROOT")
fi
COMMAND+=(
  "${SOURCE_ARGS[@]}"
  "${MOTION_ARGS[@]}"
  "${VIDEO_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)

printf '\n即將執行：\n  cd %q\n  ' "$RUN_PROJECT"
printf '%q ' "${COMMAND[@]}"
printf '\n'

if [[ $DRY_RUN == true ]]; then
  printf '\n[資訊] dry-run 完成，未開始播放。\n'
  exit 0
fi

prompt_yes_no "開始播放？" no
[[ $ANSWER == True ]] || cancel

cd -- "$RUN_PROJECT"
exec "${COMMAND[@]}"
