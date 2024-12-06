# model_training.py

from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np


def train_and_evaluate_model(
    X_train_fe: pd.DataFrame,
    y_train: pd.Series,
    X_val_fe: pd.DataFrame,
    y_val: pd.Series,
    use_optuna: bool = False,
) -> tuple:
    """
    訓練並評估模型，返回最佳參數。

    :param X_train_fe: 訓練集特徵的 DataFrame
    :param y_train: 訓練集標籤的 Series
    :param X_val_fe: 驗證集特徵的 DataFrame
    :param y_val: 驗證集標籤的 Series
    :param use_optuna: 是否使用 Optuna 進行超參數調整，預設為 False
    :return: 包含最佳參數的元組 (best_params_xgb, best_params_lgb, best_params_cat)
    """
    if use_optuna:
        best_params_xgb = tune_xgb_params(X_train_fe, y_train, X_val_fe, y_val)
        best_params_lgb = tune_lgb_params(X_train_fe, y_train, X_val_fe, y_val)
        best_params_cat = tune_cat_params(X_train_fe, y_train, X_val_fe, y_val)
    else:
        # 使用預設或之前找到的最佳參數
        best_params_xgb = {
            "n_estimators": 8000,
            "learning_rate": 0.1,
            "max_depth": 12,
            "eval_metric": ["mae", "rmse"],
            "tree_method": "hist",
        }
        best_params_lgb = {
            "n_estimators": 8000,
            "learning_rate": 0.1,
            "max_depth": 12,
            "metric": ["mae", "rmse"],
        }
        best_params_cat = {
            "iterations": 8000,
            "learning_rate": 0.1,
            "depth": 12,
            "loss_function": "RMSE",
        }

    print("已獲取最佳參數。")
    return best_params_xgb, best_params_lgb, best_params_cat


def tune_xgb_params(
    X_train_fe: pd.DataFrame, y_train: pd.Series, X_val_fe: pd.DataFrame, y_val: pd.Series
) -> dict:
    """
    使用 Optuna 調整 XGBoost 的超參數。

    :param X_train_fe: 訓練集特徵的 DataFrame
    :param y_train: 訓練集標籤的 Series
    :param X_val_fe: 驗證集特徵的 DataFrame
    :param y_val: 驗證集標籤的 Series
    :return: 最佳參數的字典
    """

    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 10000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 13),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "tree_method": "hist",
        }

        model = xgb.XGBRegressor(random_state=42, **params)

        model.fit(
            X_train_fe,
            y_train,
            eval_set=[(X_val_fe, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )

        y_val_pred = model.predict(X_val_fe)
        # 確保 y_val_pred 為一維陣列
        y_val_pred = np.asarray(y_val_pred).flatten()
        mae = mean_absolute_error(y_val, y_val_pred)

        return float(mae)

    study_xgb = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study_xgb.optimize(objective_xgb, n_trials=50)
    best_params = study_xgb.best_params
    print("XGBoost 的超參數調整已完成。")
    print("最佳參數：", best_params)
    return best_params


def tune_lgb_params(
    X_train_fe: pd.DataFrame, y_train: pd.Series, X_val_fe: pd.DataFrame, y_val: pd.Series
) -> dict:
    """
    使用 Optuna 調整 LightGBM 的超參數。

    :param X_train_fe: 訓練集特徵的 DataFrame
    :param y_train: 訓練集標籤的 Series
    :param X_val_fe: 驗證集特徵的 DataFrame
    :param y_val: 驗證集標籤的 Series
    :return: 最佳參數的字典
    """

    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 10000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }

        model = lgb.LGBMRegressor(random_state=42, **params)

        model.fit(
            X_train_fe,
            y_train,
            eval_set=[(X_val_fe, y_val)],
        )

        y_val_pred = model.predict(X_val_fe)
        # 確保 y_val_pred 為一維陣列
        y_val_pred = np.asarray(y_val_pred).flatten()
        mae = mean_absolute_error(y_val, y_val_pred)

        return float(mae)

    study_lgb = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study_lgb.optimize(objective_lgb, n_trials=50)
    best_params = study_lgb.best_params
    print("LightGBM 的超參數調整已完成。")
    print("最佳參數：", best_params)
    return best_params


def tune_cat_params(
    X_train_fe: pd.DataFrame, y_train: pd.Series, X_val_fe: pd.DataFrame, y_val: pd.Series
) -> dict:
    """
    使用 Optuna 調整 CatBoost 的超參數。

    :param X_train_fe: 訓練集特徵的 DataFrame
    :param y_train: 訓練集標籤的 Series
    :param X_val_fe: 驗證集特徵的 DataFrame
    :param y_val: 驗證集標籤的 Series
    :return: 最佳參數的字典
    """

    def objective_cat(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 1000, 10000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "depth": trial.suggest_int("depth", 3, 13),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "loss_function": "RMSE",
            "eval_metric": "MAE",
            "verbose": False,
        }

        model = CatBoostRegressor(random_state=42, **params)

        model.fit(
            X_train_fe,
            y_train,
            eval_set=(X_val_fe, y_val),
            early_stopping_rounds=50,
        )

        y_val_pred = model.predict(X_val_fe)
        # 確保 y_val_pred 為一維陣列
        y_val_pred = np.asarray(y_val_pred).flatten()
        mae = mean_absolute_error(y_val, y_val_pred)

        return float(mae)

    study_cat = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study_cat.optimize(objective_cat, n_trials=50)
    best_params = study_cat.best_params
    print("CatBoost 的超參數調整已完成。")
    print("最佳參數：", best_params)
    return best_params


def train_stacking_model(
    X_train_fe: pd.DataFrame,
    y_train: pd.Series,
    X_val_fe: pd.DataFrame,
    y_val: pd.Series,
    best_params_xgb: dict,
    best_params_lgb: dict,
    best_params_cat: dict,
) -> StackingRegressor:
    """
    定義並訓練堆疊模型。

    :param X_train_fe: 訓練集特徵的 DataFrame
    :param y_train: 訓練集標籤的 Series
    :param X_val_fe: 驗證集特徵的 DataFrame
    :param y_val: 驗證集標籤的 Series
    :param best_params_xgb: XGBoost 的最佳參數字典
    :param best_params_lgb: LightGBM 的最佳參數字典
    :param best_params_cat: CatBoost 的最佳參數字典
    :return: 已訓練的堆疊模型
    """
    # 定義基礎模型
    xgb_model = xgb.XGBRegressor(random_state=42, **best_params_xgb)
    lgb_model = lgb.LGBMRegressor(random_state=42, **best_params_lgb)
    cat_model = CatBoostRegressor(random_state=42, verbose=0, **best_params_cat)

    print("已定義基礎模型。")

    # 定義堆疊模型（Stacking）
    estimators = [
        ("xgb", xgb_model),
        ("lgb", lgb_model),
        ("cat", cat_model),
    ]

    # 使用 XGBoost 作為最終的元學習器
    stack_model = StackingRegressor(
        estimators=estimators,
        final_estimator=xgb.XGBRegressor(random_state=42),
        n_jobs=-1,
    )

    print("已定義堆疊模型。")

    # 訓練堆疊模型
    stack_model.fit(X_train_fe, y_train)
    print("堆疊模型已訓練完成。")

    # 驗證堆疊模型
    y_val_pred = stack_model.predict(X_val_fe)
    y_val_pred = np.asarray(y_val_pred).flatten()
    mae = mean_absolute_error(y_val, y_val_pred)
    print(f"驗證集的 MAE: {mae}")

    return stack_model


# def predict_test_data(
#     X_train_fe: pd.DataFrame,
#     y_train: pd.Series,
#     X_val_fe: pd.DataFrame,
#     y_val: pd.Series,
#     test_data_features_fe: pd.DataFrame,
#     best_params_xgb: dict,
#     best_params_lgb: dict,
#     best_params_cat: dict,
# ) -> pd.Series:
#     """
#     訓練模型並對測試資料進行預測。

#     :param X_train_fe: 訓練集特徵的 DataFrame
#     :param y_train: 訓練集標籤的 Series
#     :param X_val_fe: 驗證集特徵的 DataFrame
#     :param y_val: 驗證集標籤的 Series
#     :param test_data_features_fe: 測試資料特徵的 DataFrame
#     :param best_params_xgb: XGBoost 的最佳參數字典
#     :param best_params_lgb: LightGBM 的最佳參數字典
#     :param best_params_cat: CatBoost 的最佳參數字典
#     :return: 測試資料的預測結果 Series
#     """
#     stack_model = train_stacking_model(
#         X_train_fe, y_train, X_val_fe, y_val, best_params_xgb, best_params_lgb, best_params_cat
#     )


#     # 對測試集進行預測
#     y_test_pred = stack_model.predict(test_data_features_fe)
#     y_test_pred = np.asarray(y_test_pred).flatten()
#     return pd.Series(y_test_pred)
def predict_test_data(
    X_train_fe: pd.DataFrame,
    y_train: pd.Series,
    X_val_fe: pd.DataFrame,
    y_val: pd.Series,
    test_data_features_fe: pd.DataFrame,
    best_params_xgb: dict,
    best_params_lgb: dict,
    best_params_cat: dict,
) -> tuple:
    """
    訓練模型並對測試資料進行預測，同時回傳堆疊模型和預測結果。

    :return: (stack_model, y_test_pred)
    """
    stack_model = train_stacking_model(
        X_train_fe, y_train, X_val_fe, y_val, best_params_xgb, best_params_lgb, best_params_cat
    )

    # 對測試集進行預測
    y_test_pred = stack_model.predict(test_data_features_fe)
    y_test_pred = np.asarray(y_test_pred).flatten()
    return stack_model, pd.Series(y_test_pred)
