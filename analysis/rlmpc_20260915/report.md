分析日期：2026-09-15。目標依現有 G23 實機部署介面判斷。

在本次抽測的四個 checkpoint 中，若以平地低速第一輪實機 A/B 的候選來選，我偏向 **2026-09-10_11-11-39/model_2400.pt**：XY 追蹤比最後一版好，原始 action 裁切也較少。這是暫定選擇，並非已證明它在實機較安全。**model_5399.pt 的停止姿態最好**，如果主要目標是停後站姿，它是值得保留的另一候選。首次實機驗證仍應先建立純 MPC 基準，再比較 residual policy。

目前沒有任何一版通過本次完整地形停止測試，因此不建議直接放行粗糙地形實機運行。

比較對象與來源

最新 run 是 `logs/rsl_rl/g23_rl_mpc_foot_state_history_v3/2026-09-10_11-11-39`，由 `2026-09-10_10-51-22/model_400.pt` 恢復訓練，最後 checkpoint 為 `model_5399.pt`。抽測同一 run 的 700、2400、4000、5399，另外加入零 residual 的純 MPC；未逐一掃描所有 checkpoint，不能稱為全 checkpoint 最佳解。之前分析中的 2026-09-09 run 已不在本地 checkpoint 目錄，沒有沿用它的數據排名。

共同介面為 375 維觀測（5 × 75，含 80 ms 歷史）、8 維 XY 落腳 residual，policy/MPC 50 Hz、controller 100 Hz、步態 0.5 秒，無地形掃描。使用目前 mjlab/thirdparty controller、SyncAI formulation、qpOASES、完整 actor normalizer、deterministic actor，啟用目前的對稱 residual 投影與兩步停止控制。這些控制器功能對所有候選及純 MPC相同，其改善不能全歸因於 policy。

平地結果

每個候選使用同樣 seed 42、同樣初始姿態，測試七種 `[vx,vy,wz]`：`[0.2,0,0]`、`[-0.2,0,0]`、`[0,0.1,0]`、`[0,-0.1,0]`、`[0,0,0.3]`、`[0,0,-0.3]`、`[0.2,0,0.3]`。每次站 2 秒、走 4 秒、停 3 秒；追蹤及裁切取行走後 2 秒，停止姿態取停止後半段 1.5 秒。關節 RMSE 以 12 個關節相對 default pose 計算。下表對七個指令等權平均，XYZ MAE 分別保留單位，不混成任意加权總分。

| 模型 | vx MAE m/s | vy MAE m/s | yaw MAE rad/s | 停姿平均 RMSE ° | 停姿最差 RMSE ° | 行走 raw action 裁切 |
|---|---:|---:|---:|---:|---:|---:|
| pure_mpc | 0.0222 | 0.0172 | 0.0824 | 2.12 | 2.39 | 0.0% |
| 700 | 0.0171 | 0.0163 | 0.0841 | 2.12 | 2.81 | 28.5% |
| 2400 | 0.0175 | 0.0182 | 0.0852 | 1.81 | 2.26 | 16.5% |
| 4000 | 0.0219 | 0.0220 | 0.0820 | 1.96 | 2.37 | 24.0% |
| 5399 | 0.0188 | 0.0221 | 0.0853 | 1.56 | 2.03 | 27.5% |

35 次平地流程全數完成，停止統計區間皆為 STAND 且四腳接觸比例 100%，native solver failure count 皆為 0。這是一個固定初始化的短程測試，並非 35 個独立隨機 seed。

相較 5399，2400 的 vx MAE 約小 6.7%、vy MAE 約小 17.8%，yaw 幾乎相同；但停姿 RMSE 多約 0.25°。5399 的停姿比純 MPC更接近 default pose；700 的 XY 追蹤也很好，卻有較高裁切和較差的最壞停姿。4000 的訓練 reward 較高，實測並沒有全面領先，因此不按 reward 排第一。

Raw action 超出 [-1,1] 的比例只是網路輸出特性，不是最終足端越界比例，也不能直接證明馬達更平順或更安全；還有 clipping、對稱投影、指令相關限幅及 swing latch。沒有實測 torque margin 或 action jerk，所以不對它們作結論。

最高地形結果

Level 5 為設定的最高地形難度（高度峰谷差上限 8 cm）；三個環境分別為 flat、random_rough、low_wave。每個模型測試前進、橫移、旋轉三種正向單軸指令，仍採 2/4/3 秒流程，共 15 組、每組 3 個環境。任一環境終止就停止整組，其他環境剩餘時間不算已驗證；以下是明確觸發的終止事件，不是抽樣成功率。

| 模型 | 前進組 | 橫移組 | 旋轉組 |
|---|---|---|---|
| 純 MPC | 無終止，但 low_wave 未停回 STAND | 完成 | low_wave 停止 2.38 s 終止 |
| 700 | 無終止，但 rough/wave 未停回 STAND | 完成 | low_wave 停止 2.38 s 終止 |
| 2400 | low_wave 停止 2.28 s 終止 | 完成 | low_wave 停止 1.58 s 終止 |
| 4000 | low_wave 停止 0.74 s 終止 | 完成 | random_rough 停止 1.50 s 終止 |
| 5399 | low_wave 停止 2.08 s 終止 | 完成 | random_rough 停止 1.92 s 終止 |

所有明確提前終止都在停止階段，未見 native solver 報告失敗。這將停止控制及接觸／地形配合列為優先排查對象，但目前未記錄每次終止的具體 term，不能直接斷言根因。未終止也不代表穩定：純 MPC 的前進 low_wave 在停止尾段不在 STAND、四腳同時接觸僅 4%。不平地的 default joint pose 本來可能不適合地形，所以不能單憑大關節 RMSE 判定失敗。

固定地形 seed 42、固定起點，未涵蓋多地形 seed、擾動、馬達延遲、感測噪聲、完整速度上限或長時間循環。為避免提前終止導致不同長度的資料造成偏差，不使用地形平均追蹤值決定排名。

訓練趨勢

在各 checkpoint 前 100 iteration 範圍內的平均：reward 在 4000 約 101.14，5399 約 100.29；平均地形等級約 1.20 → 0.86；yaw 日誌誤差約 0.129 → 0.148。最後一版未見訓練優勢全面延續。日誌速度誤差經過命令時窗正規化，不是本次實測 MAE。停止 penalty 的訓練趨勢與固定平地停姿排名也不同，顯示不能只用 reward 選模。[訓練曲線](training_curves.png)、[checkpoint 時窗數據](checkpoint_training_windows.json)。

部署核對與實機限制

目前 `quadruped_robot_simulation/policy/mpc/mjlab_foot_state_history_v3.onnx` 已經是 **model_5399.pt**，其 metadata SHA256 與 checkpoint 一致：`f36090fb80d68e0aebc56ba4a82eb797f1365e532fb942682868c9cf2b4e980e`。20 組依 normalizer 均值／標準差產生的輸入，ONNX 與 PyTorch 最大絕對輸出差約 `2.98e-7`。這驗證匯出數值一致，沒有驗證完整部署控制迴圈或實機資料分布。

部署 Rust runner 支援此 375→8 介面及對稱／停止流程，但其 QP solver 是 Clarabel，MJLab 是 qpOASES，仍需以同 checkpoint 核對實測觀測、接觸估測與控制時序。`config/g2_rlmpc_foot_state_history.env` 範例預設 `G2_RLMPC_BARE=1`，會使用純 MPC；這不等於已確認使用者當前實機程序的環境變數。

目前訓練設定移除了 push、摩擦、encoder bias、base CoM 的 domain randomization，完整觀測歷史也沒有加入 corruption；訓練仍有初始位置、高度及 yaw 的隨機化，並非完全沒有隨機性。命令訓練只抽單軸，本次混合命令通過一例不能當作混合操作泛化證明。

建議的選擇是：先在部署迴圈完成純 MPC 平地基準，再把 **2400 作為第一個低速 residual A/B 候選，5399 作為停姿對照**。若目標是粗糙地形、推擾或無保護的連續實機運行，目前應先修正／驗證停止行為，加入實機誤差來源的訓練與測試，再重新選模。這次沒有替換任何部署 ONNX、修改 controller 或啟動實機。

重現與附件

於 mjlab_test 根目錄：

```bash
MPLCONFIGDIR=/tmp/mjlab-mpl UV_CACHE_DIR=/tmp/mjlab-uv-cache uv run --no-sync python analysis/rlmpc_20260915/evaluate.py
MPLCONFIGDIR=/tmp/mjlab-mpl UV_CACHE_DIR=/tmp/mjlab-uv-cache uv run --no-sync python analysis/rlmpc_20260915/evaluate.py --models pure_mpc 2400 4000 5399 --levels 5 --terrain-bank --positive-axes-only --output terrain.json
MPLCONFIGDIR=/tmp/mjlab-mpl UV_CACHE_DIR=/tmp/mjlab-uv-cache uv run --no-sync python analysis/rlmpc_20260915/evaluate.py --models 700 --levels 5 --terrain-bank --positive-axes-only --output terrain_700.json
UV_CACHE_DIR=/tmp/mjlab-uv-cache uv run --no-sync python analysis/rlmpc_20260915/summarize.py
```

[完整平地數據](screening.json)、[地形數據](terrain.json)、[700 地形數據](terrain_700.json)、[汇總](summary.json)、[部署核對](deployment_check.json)、[來源 commit 與 hash](provenance.json)、[訓練與 metadata inventory](inventory.json)。
