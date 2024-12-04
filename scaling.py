# scaling.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_data(
    X_train_fe: pd.DataFrame, X_val_fe: pd.DataFrame, test_data_features_fe: pd.DataFrame
) -> tuple:
    """
    使用 MinMaxScaler 對資料進行縮放。

    :param X_train_fe: 訓練集特徵的 DataFrame
    :param X_val_fe: 驗證集特徵的 DataFrame
    :param test_data_features_fe: 測試資料特徵的 DataFrame
    :return: 包含縮放後的 X_train_fe, X_val_fe, test_data_features_fe 的元組
    """
    scaler = MinMaxScaler()
    X_train_fe_col = X_train_fe.columns

    # 對訓練和驗證集進行縮放
    X_train_fe_scaled = scaler.fit_transform(X_train_fe)
    X_val_fe_scaled = scaler.transform(X_val_fe)

    # 確保測試集的特徵順序與訓練集一致，並進行縮放
    test_data_features_fe = test_data_features_fe[X_train_fe_col]
    test_data_features_fe_scaled = scaler.transform(test_data_features_fe)

    # 將縮放後的資料轉換回 DataFrame，並保留列名
    X_train_fe_scaled = pd.DataFrame(X_train_fe_scaled, columns=X_train_fe_col)
    X_val_fe_scaled = pd.DataFrame(X_val_fe_scaled, columns=X_train_fe_col)
    test_data_features_fe_scaled = pd.DataFrame(
        test_data_features_fe_scaled, columns=X_train_fe_col
    )

    return X_train_fe_scaled, X_val_fe_scaled, test_data_features_fe_scaled
