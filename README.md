# aicup2024
# 環境配置
本專案原先使用 `miniconda` 作為環境管理工具，但實驗過程安裝套件過多，直接匯出conda提供的 `.yml` 文件會讓安裝過程產生衝突，因此僅提供 `requirements.txt` 作為套件清單。
可自行使用任一環境管理工具，但需確認 Python 版本要是 `3.11.10`。（經測試 `3.11.11` 也能正常運行）

## 安裝依賴套件
```bash
pip install -r requirements.txt
```

## 注意事項

> **⚠️ 請注意 ⚠️**
> 1. 該環境及程式僅在本團隊之設備上測試，由於機器學習會使用大量記憶體，若記憶體較少或是CPU等級較低，會大幅影響程式執行速度，下方將提供本團隊設備及執行時間供參考。
> 
> 2. 由於資料前處理並不是在實驗過程中會重複執行的一個部分，因此我們會使用 `concat_dataset.py` 來將所有原始資料(例如 `L1_Train.csv`)合併成 `concat.csv`之後主程式也只會呼叫 `concat.csv` 。
>
> 3. 在 `feature_importance.py` 中，為了顯示中文我們使用的是 `Noto Sans CJK JP` 字體，請確保運行裝置有該字體，或是替換成任一支持中文的字體，否則會造成中文字型顯示問題。

| 環境設備 | 內容 |
| --- | --- |
| OS | Ubuntu 20.04 |
| CPU | Intel i9-13900K |
| GPU | NVIDIA GeForce RTX 4090 |
| RAM | 64 GB |
| 執行時間 (使用Optuna) | 約 60 ~ 180 分鐘 |
| 執行時間 (固定參數) | 約 30 ~ 90 分鐘 |
---
# Python 程式

## 使用方法
```bash
python ./main.py
```
將會依照以下步驟逐一進行
1. 合併資料集
2. 特徵工程 (OpenFE) (可選)
3. 縮放特徵
4. 尋找最佳超參數 (Optuna) (可選)
5. 訓練及預測
6. 繪製特徵重要性圖

## 專案結構
```
.
├── LSTM                        # LSTM實驗結果
├── aicup_empty_answer.csv      # upload.csv
├── catboost_info               # catboost自動生成緩存資料夾
├── concat.csv                  # 經過資料清洗後整合完的資料
├── concat_dataset.py           # 處理原始資料集，整合成concat.csv
├── data_loader.py              # 提供讀取資料的功能
├── data_preprocessing.py       # 處理外部資料
├── feature_engineering.py      # 使用OpenFE做特徵工程
├── feature_importance.py       # 繪製特徵重要性圖
├── location_azimuths.csv       # 外部資料-太陽能板方位，來自官方說明V2
├── main.py                     # 主程式，整合各模組功能
├── model_training.py           # 定義模型、訓練、預測
├── openfe_tmp_data.feather     # OpenFE自動生成緩存檔案
├── scaling.py                  # 資料正規化
├── solar_angles_10min.csv      # 外部資料-仰角
├── solar_azimuths.csv          # 外部資料-方位角
├── submission.csv              # 最終上傳的答案
├── submission.py               # 將答案與序號整合，製作submission.csv
├── test_data_preprocessing.py  # 處理無答案的csv，加入方位角、仰角等資料整合成可以進行預測的格式
├── utils.py                    # 將序號轉換為時間
└── requirements.txt            # Python 套件清單
```