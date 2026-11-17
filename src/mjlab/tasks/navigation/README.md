# 醫院導航任務（rl_agv 分支）

從 [mujoco-rl-nav](https://github.com/otr-ebla/MuJoCo_HumanAware_MobileRobot_RLNavigation)（SB3 + 單一 MuJoCo 環境）移植到 mjlab（MuJoCo-Warp GPU 並行 + rsl_rl PPO）的 LiDAR 導航任務。這份文件是給之後接手修改的人（或 Claude）看的交接說明。

## 任務

| Task ID | 內容 |
|---|---|
| `Mjlab-Navigation-Hospital` | 醫院地圖、無行人 |
| `Mjlab-Navigation-Hospital-Humans` | 同上，加上最多 20 位社會力模型行人 |

兩個任務的觀測維度相同、共用實驗資料夾 `logs/rsl_rl/navigation_hospital/`，所以無行人訓練出的權重可以直接 resume 到有行人任務。

## 檔案

| 檔案 | 內容 |
|---|---|
| `__init__.py` | 註冊兩個任務 |
| `navigation_env_cfg.py` | 環境設定（場景、觀測、動作、獎勵權重、終止條件）與 PPO 設定。**要調參數先看這裡** |
| `assets.py` | 地圖、機器人、行人的 MjSpec；半徑、LiDAR 高度等常數 |
| `mdp/actions.py` | `UnicycleAction`（機器人）、`PedestrianAction`（行人，動作維度 0） |
| `mdp/commands.py` | `NavigationCommand`：reset 時抽情境、擺放機器人與行人、保存目標 |
| `mdp/hsfm.py` | Headed Social Force Model 的 PyTorch 批次版 |
| `mdp/geometry.py` | `obstacle_distance`：機器人中心到最近障礙物的解析距離 |
| `mdp/lidar.py` | `PlanarLidarPatternCfg`：給 `RayCastSensor` 用的 360° 平面射線 |
| `mdp/observations.py`、`rewards.py`、`terminations.py` | 觀測、獎勵、終止函式 |
| `data/hospital_obstacles.csv` | 61 個軸對齊方塊（43 牆、16 桌、2 個只給行人用的虛擬障礙） |
| `data/scenarios.npz` | 17 個情境 × 2048 組預先取樣的起點/目標/行人配置 |
| `scripts/convert_assets.py` | 從 mujoco-rl-nav 重新產生 `data/` 的轉換工具 |
| `tests/test_navigation_task.py`（repo 根目錄） | 7 項測試 |

## 設計重點

- **全部是運動學**：機器人（x、y、yaw 三個關節）和行人（每人三個關節）都是直接寫關節位置，沒有接觸、沒有致動器。因此 `timestep=0.25`、`decimation=1`、關閉 contact，一個控制步只跑一次物理步。
- **所有環境共用世界原點**（`env_spacing=0`）：每個環境在自己的 world 裡用同一張地圖，關節座標就是世界座標。
- **行人放在 action manager**：`PedestrianAction` 不吃 policy 輸出，只是為了每個物理步跟著機器人一起前進。每個控制步內部以 0.025 秒積分 10 次。
- **碰撞與接近懲罰不用 LiDAR**：mjlab 的 step 順序是「動作 → 終止 → 獎勵 → reset → sense → 觀測」，算獎勵時 LiDAR 還是上一步的資料。所以碰撞（≤ 0.2 m）與接近懲罰用 `obstacle_distance` 從關節位置直接算，等同理想 LiDAR 的最小讀值。LiDAR 只用在觀測。
- **進度獎勵的 `prev_distance`**：存在 `NavigationCommand`，在 `_update_command`（獎勵算完之後）更新。
- **情境抽樣**：`adaptive_sampling=True` 時依各情境失敗率加權（原專案的做法），所以訓練時 log 的 `success_rate` 會偏低；play 時改為均勻抽樣。
- **未使用的行人**：原情境把不用的行人放在地圖外（甚至 x = −1e6）。轉換時起點在地圖外的一律標為未啟用，並停在 (−150, −150) 附近。若把極大座標寫進 qpos，整個 world 會變成 NaN。

## 與原專案的差異

- 演算法是 rsl_rl PPO，不是 SB3 的 TQC/SAC。
- 獎勵整體乘 0.1（`_REWARD_SCALE`）：成功 +20、碰撞 −7，相對比例不變；進度獎勵係數固定 0.06（原本隨成功率在 0.04–0.08 變動）。`scale_rewards_by_dt=False`。
- 動作：policy 輸出 `[-1, 1]`，線速度映射到 0–1 m/s，角速度 ±1 rad/s。
- LiDAR 最大距離 20 m，沒打到回報 20；觀測有縮放並開啟 obs normalization。
- 行人速度 1.0 m/s（原專案 `env_config.py` 寫 0.001，但程式內警告顯示 1.0 才是預期值）。
- 行人換目標的距離門檻是 0.3 m（原本 0.1 m）。
- 行人的障礙物力用「4 m 內的方塊」取平均，近似原本的格點查表。
- 訓練時行人不避讓機器人、play 時會避讓（`react_to_robot=play`），與原專案一致。

## 環境與指令

需要 `uv`（`pip install --user uv`，執行檔在 `~/.local/bin`），在 repo 根目錄 `uv sync`。

```sh
# 無行人：200 次迭代（RTX 3080 約 5 分鐘）成功率 99%+
uv run train Mjlab-Navigation-Hospital --env.scene.num-envs 2048 --agent.logger tensorboard

# 有行人：從無行人權重接續（<run> 是 logs/rsl_rl/navigation_hospital/ 下的資料夾名）
uv run train Mjlab-Navigation-Hospital-Humans --env.scene.num-envs 2048 \
  --agent.logger tensorboard --agent.resume True --agent.load-run <run>

# 測試與檢查
uv run pytest tests/test_navigation_task.py
uv run ruff format && uv run ruff check --fix && uv run ty check

# 重新產生 data/（只需要 numpy）
uv run python src/mjlab/tasks/navigation/scripts/convert_assets.py /path/to/mujoco-rl-nav
```

## 目前狀態與已知問題

- 有行人任務**從零訓練很慢**（約 150 次迭代成功率只有 5–8%，機器人傾向停著不動）；從無行人權重接續約 110 次迭代可到約 49% 且仍在上升。**尚未做過完整訓練**。
- 少數情境樣本在 reset 當下機器人就與行人或障礙物重疊，該回合會立刻以碰撞結束（原專案也有）。
- 沒有在 viewer 裡目視檢查過畫面。
- 這台開發機的 Node 太舊，`pyright` 跑不起來；`ruff`、`ty` 通過。
- `data/` 是從 mujoco-rl-nav 轉出來的，其授權與 mjlab 的 Apache-2.0 不同，對外發布前要留意。
