#!/usr/bin/env bash
# Interactive frontend for the fixed 375 -> 8 MJLab RLMPC exporter.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
LOG_ROOT="$PROJECT_DIR/logs/rsl_rl/g23_rl_mpc_foot_state_history_v3"
DEFAULT_OUTPUT="$PROJECT_DIR/../quadruped_robot_simulation/policy/mpc/mjlab_foot_state_history_v3.onnx"
DRY_RUN=0
ANSWER=""
SELECTED=""

usage() {
  cat <<'EOF'
互動式匯出 RLMPC ONNX（375 維輸入 → 8 維落腳點輸出）。

用法：
  ./scripts/export_rl_mpc_policy.sh
  ./scripts/export_rl_mpc_policy.sh --dry-run
  ./scripts/export_rl_mpc_policy.sh --log-root /path/to/training/runs

選擇訓練 run → checkpoint → 輸出路徑 → 匯出。
Enter 選擇預設值；選單輸入 0 可手動指定 checkpoint；q 或 Ctrl+C 取消。
--dry-run 只顯示匯出指令，不執行、不修改輸出檔。
相對路徑以啟動腳本時的工作目錄為準；路徑請直接輸入，不加引號。
EOF
}

die() { printf '錯誤：%s\n' "$*" >&2; exit 1; }
cancel() { printf '\n已取消匯出。\n'; exit 0; }
trap cancel INT

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --log-root)
      (($# >= 2)) || die '--log-root 缺少路徑。'
      LOG_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "不支援的參數：$1（使用 --help 查看說明）" ;;
  esac
done

ask() {
  printf '%s' "$1"
  IFS= read -r ANSWER || cancel
  [[ "$ANSWER" != q && "$ANSWER" != Q ]] || cancel
}

# No eval: spaces and shell metacharacters in paths remain literal arguments.
absolute_path() {
  case "$1" in
    '~/'*) REPLY="$HOME/${1:2}" ;;
    /*) REPLY="$1" ;;
    *) REPLY="$PWD/$1" ;;
  esac
}

choose() {
  local title="$1" index i
  shift
  local items=("$@")
  printf '\n%s（最新在前，Enter 選 1）\n' "$title"
  for i in "${!items[@]}"; do
    printf '  %d) %s\n' "$((i + 1))" "${items[i]##*/}"
  done
  printf '  0) 手動輸入 checkpoint 路徑\n'
  while true; do
    ask '請選擇 [1]，或 q 取消：'
    index="${ANSWER:-1}"
    if [[ "$index" =~ ^[0-9]{1,6}$ ]]; then
      index=$((10#$index))
      if ((index == 0)); then SELECTED=""; return; fi
      if ((index <= ${#items[@]})); then
        SELECTED="${items[index - 1]}"
        return
      fi
    fi
    printf '請輸入清單中的編號。\n'
  done
}

absolute_path "$LOG_ROOT"
LOG_ROOT="$REPLY"
printf 'RLMPC ONNX 匯出：375 → 8，包含觀測正規化與推論一致性檢查。\n'
printf '訓練紀錄：%s\n' "$LOG_ROOT"
shopt -s nullglob
runs=()
if [[ -d "$LOG_ROOT" ]]; then
  while IFS= read -r -d '' run; do
    checkpoints=("$run"/model_*.pt)
    if ((${#checkpoints[@]})); then runs+=("$run"); fi
  done < <(find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -zr)
fi

checkpoint=""
if ((${#runs[@]})); then
  choose '選擇訓練 run' "${runs[@]}"
  if [[ -n "$SELECTED" ]]; then
    mapfile -d '' -t checkpoints < <(
      find "$SELECTED" -maxdepth 1 -type f -name 'model_*.pt' -print0 | sort -zVr
    )
    ((${#checkpoints[@]})) || die '此 run 沒有可讀取的 checkpoint。'
    choose '選擇 checkpoint' "${checkpoints[@]}"
    checkpoint="$SELECTED"
  fi
else
  printf '沒有找到 RLMPC checkpoint，請手動指定。\n'
fi

while [[ ! -f "$checkpoint" ]]; do
  ask 'Checkpoint 完整路徑（q 取消）：'
  [[ -n "$ANSWER" ]] || continue
  absolute_path "$ANSWER"
  checkpoint="$REPLY"
  [[ -f "$checkpoint" ]] || printf '找不到檔案：%s\n' "$checkpoint"
done

while true; do
  printf '\n預設 ONNX：%s\n' "$DEFAULT_OUTPUT"
  ask '輸出路徑 [Enter 使用預設]：'
  absolute_path "${ANSWER:-$DEFAULT_OUTPUT}"
  output="$REPLY"
  [[ "$output" == *.onnx ]] || { printf '輸出檔名必須以 .onnx 結尾。\n'; continue; }
  [[ ! -d "$output" ]] || { printf '輸出路徑是資料夾，請指定檔名。\n'; continue; }
  break
done

command=(uv run --no-sync python "$SCRIPT_DIR/export_rl_mpc_policy.py" "$checkpoint" "$output")
if [[ -e "$output" || -L "$output" ]]; then
  ask '輸出檔已存在，是否覆蓋？[y/N]：'
  case "$ANSWER" in y|Y|yes|YES) command+=(--force) ;; *) cancel ;; esac
fi
printf '\nCheckpoint：%s\n輸出 ONNX：%s\n執行指令：' "$checkpoint" "$output"
printf '%q ' "${command[@]}"
printf '\n'
if ((DRY_RUN)); then
  printf 'Dry run 完成，未執行匯出。\n'
  exit 0
fi
command -v uv >/dev/null 2>&1 || die '找不到 uv，請先安裝 uv 並準備 mjlab 環境。'
cd -- "$PROJECT_DIR"
if "${command[@]}"; then
  printf '\n匯出與推論一致性檢查成功：%s\n' "$output"
else
  status=$?
  printf '\n匯出失敗（exit %d），請查看上方錯誤訊息。\n' "$status" >&2
  exit "$status"
fi
