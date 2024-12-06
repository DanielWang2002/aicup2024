- 本次實驗是通過 LSTM+MLP 與 XGBoost 訓練時聯合計算損失值，並且根據不同地區分出不同類別去做訓練出該類別的模型，屬於集成模型的預測方法，並且預測方法同時使用 LSTM+MLP 與 XGBoost 預測，並將結果加總平均做輸。

- 套件安裝

  ```
  pip install pandas
  pip install numpy
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install scikit-learn
  pip install xgboost
  ``` 

- 模型輸入
    - `Sunlight(Lux)_normalized`：將原始資料集 'Sunlight(Lux)' 做 StandardScaler 正規化後的特徵，加速網路收斂速度
    - `仰角`：通過[中央氣象署](https://www.cwa.gov.tw/V8/C/K/astronomy_day.html)爬取的太陽仰角，經過公式轉換得到每十分鐘的太陽仰角度數
    - `Time_sin`：將時間通過以下公式做編碼，使時間成為具有連續性的特徵
            np.sin(2 * np.pi * data['MinutesOfDay'] / (24 * 60))

- 模型輸出
    - `Power(mW)`：預測的電力功率，單位為 mW