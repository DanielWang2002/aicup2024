# aicup2024
# 環境配置
本次競賽中使用 `miniconda` 作為環境管理工具  
以下指令皆需安裝 `anaconda` / `miniconda`  
## 1. 配置conda env及使用套件
```bash
conda env create -f ./aicup.yml
```
## 2. 啟用conda env
```bash
conda activate aicup
```
## 3. 查看版本確認是否建置及啟用成功
```bash
python -V
pip -V
```
輸出應類似於下方
```bash
Python 3.11.10
pip 24.2 from /home/danielwang/miniconda3/envs/aicup/lib/python3.11/site-packages/pip (python 3.11)
```
# 使用方法
```bash
python ./main.py
```
# 專案結構
```
.
├── LSTM                        # LSTM實驗結果
├── aicup_empty_answer.csv      # upload.csv
├── catboost_info               # catboost自動生成緩存資料夾
├── concat.csv                  # 經過資料清洗後整合完的資料
├── data_loader.py              # 提供讀取資料的功能
├── data_preprocessing.py       # 處理外部資料
├── feature_engineering.py      # 使用OpenFE做特徵工程
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
└── utils.py                    # 將序號轉換為時間
```