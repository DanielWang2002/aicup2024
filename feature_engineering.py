# feature_engineering.py

import multiprocessing

import pandas as pd
from openfe import OpenFE, transform


def perform_feature_engineering(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    test_data_features: pd.DataFrame,
    n_features: int = 5,
) -> tuple:
    """
    使用 OpenFE 進行特徵工程。

    :param X_train: 訓練集特徵的 DataFrame
    :param y_train: 訓練集標籤的 Series
    :param X_val: 驗證集特徵的 DataFrame
    :param test_data_features: 測試資料特徵的 DataFrame
    :param n_features: 要選取的特徵數量，預設為 5
    :return: 包含 X_train_fe, X_val_fe, test_data_features_fe 的元組
    """
    cpu_count = multiprocessing.cpu_count()
    ofe = OpenFE()

    # 使用 OpenFE 生成新特徵
    features = ofe.fit(
        data=X_train,
        label=y_train,
        n_jobs=cpu_count,
        seed=42,
        verbose=False,
        feature_boosting=True,
        stage2_metric="permutation",
        task="regression",
        stage2_params={"verbose": -1},
    )
    # 只取前 n_features 個特徵
    features = features[:n_features]

    # 將新特徵應用於訓練、驗證和測試資料集
    X_train_fe, X_val_fe = transform(X_train, X_val, features, n_jobs=cpu_count)
    _, test_data_features_fe = transform(X_train, test_data_features, features, n_jobs=cpu_count)

    # 確保測試資料集包含所有訓練資料集的特徵
    missing_cols = set(X_train_fe.columns) - set(test_data_features_fe.columns)
    for col in missing_cols:
        test_data_features_fe[col] = 0

    # 對齊特徵順序
    test_data_features_fe = test_data_features_fe[X_train_fe.columns]

    return X_train_fe, X_val_fe, test_data_features_fe
