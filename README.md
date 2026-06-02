# ROGII Wellbore Geology Prediction

本專案用於 Kaggle 競賽 ROGII Wellbore Geology Prediction，目標是預測水平井在 Prediction Start (PS) 之後的 TVT。

## 目前狀態 (2026-06-02)

- 目前最佳基準: lgbm_final_reg_train.py (v13_reg 參數)
- 本地驗證: RMSE 15.08 (row-weighted), 11.95 (per-well mean)
- Public LB: 12.269
- 主力方法: 以 anchored physics 為底，再學 correction (true_tvt - anchored_physics)
- 開發中方向: spatial/formation/GR 偏移特徵與物理修正強化 (v20-v24 系列)

## 版本歷史摘要

- v1-v4: 以 dTVT 增量建模，驗證約 RMSE 17-19，累積誤差較大。
- v5: 改為直接學 correction，RMSE 下降到 15.76 (關鍵突破)。
- v6: 加入 GR matching 特徵，僅小幅改善。
- v7: 加入全域 post-PS GR 統計，效果變差 (過擬合)。
- v8: anchored physics + post-PS trajectory 特徵，RMSE 15.17。
- v9-v12: 多種延伸嘗試，未超越 v8。
- v13_reg: 加強正則化 (feature/bagging fraction, child samples, L1/L2)，最佳 RMSE 15.08。
- v14-v24: KNN / formation / xcorr / spatial / grdev 等探索中，尚未穩定超越 v13_reg 主線。

## 本次整理

- 移除不需要的 output 目錄與內容。
- 移除 archive 目錄與其內容。
- 專案以訓練與推論主流程腳本為主，實驗性腳本可視需求再重建。

## 目前建議主流程

1. 訓練最終模型
   - python lgbm_final_reg_train.py
2. 產生提交檔
   - 同一支腳本會輸出 submissions/lgbm_final_reg.csv
3. 若要針對單一 CSV 推論
   - python run_lgbm_on_test_csv.py

## 目錄說明 (重點)

- train/, test/: 競賽資料
- features/: 特徵與中間產物
- output 類目錄 (models/, predictions/, submissions/, kernel_output*/): 不納入版本管理，按需產生

## 已知最佳實務

- 用 well-level split 做驗證，不做 row-level split。
- 錨定公式使用 anchor_tvt + slope * (Z - Z_anchor)，避免截距漂移。
- pandas 3.0 以 .ffill() / .bfill() 取代 fillna(method=...)。

## 待辦

- 針對 v20-v24 系列建立統一評估表，明確比較每版 val 指標與 hard wells 行為。
- 補一支單一入口腳本，統一 train/eval/infer/submission 參數與輸出路徑。
