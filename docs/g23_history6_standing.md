# G23 History6 平地零速站姿微調

任務：`Mjlab-Velocity-Flat-SyncAI-G23-Proprio-History6-Stand`。

目標是零速度指令下回到並維持現有 HOME_KEYFRAME：四腿 HipX = 0、
HipY = -0.65、Knee = 1.3 rad。這是關節姿勢與靜止訓練，不是世界座標定點控制。

設定採平地、70% 零速指令、30% 低速移動，每 4–8 秒重新取樣，讓策略練習
移動後停止。零速時額外懲罰關節姿勢誤差及關節速度，加強四腳接觸要求，
並讓腳底滑動懲罰在零速時也生效。此階段不施加外力推擠。
play 模式全部取樣零速指令；可以手動給速度檢查走停轉換。

保留原 History6 的 actor / critic 觀測及感測器，以便載入原模型。
使用原 experiment 目錄尋找 checkpoint，續訓結果另存到含 `flat_stand` 的新 run。
原爬樓梯任務設定不變；平地微調後的策略仍需另外評估走路與爬樓梯能力。

## 續訓

先確認 `nvidia-smi` 正常，且 `.venv/bin/python -c 'import torch; print(torch.cuda.is_available())'`
輸出 `True`。在 `mjlab_test` 目錄執行：

```bash
.venv/bin/train Mjlab-Velocity-Flat-SyncAI-G23-Proprio-History6-Stand \
  --agent.resume True \
  --agent.load-run '2026-09-02_17-50-16' \
  --agent.load-checkpoint 'model_19999.pt' \
  --agent.run-name flat_stand \
  --agent.logger tensorboard \
  --agent.algorithm.entropy-coef 0.003 \
  --agent.max-iterations 1000 \
  --env.scene.num-envs 1024
```

這裡 1000 是額外訓練迭代數。續訓會載入原 optimizer 狀態。
上述獎勵權重與訓練長度是起始設定，尚未經訓練結果驗證。

## 評估

以實際新 checkpoint 路徑替換下方佔位路徑：

```bash
.venv/bin/play Mjlab-Velocity-Flat-SyncAI-G23-Proprio-History6-Stand \
  --checkpoint-file logs/rsl_rl/g23_velocity_proprio_history6/NEW_RUN/model_NEW.pt \
  --num-envs 1
```

先用原 `model_19999.pt` 跑相同任務作為基準，再比較微調模型。檢查至少
30 秒零速站立是否四腳著地、關節角接近上述目標、身體無明顯晃動或滑移，
並重複測試低速走動後停止。可搭配 TensorBoard 的
`Metrics/anti_tripod/standing_contact_rate/*` 及新增 reward 項目追蹤。
