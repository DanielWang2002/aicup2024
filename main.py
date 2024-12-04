# main.py

from xgboost import train
from data_loader import load_data
from data_preprocessing import (
    process_solar_angles,
    create_datetime_for_train_data,
    process_solar_azimuths,
    merge_solar_data,
    add_panel_azimuth,
    drop_unnecessary_columns,
    remove_datetime_column as remove_datetime_column_train,
)
from test_data_preprocessing import (
    convert_test_data_index,
    add_panel_azimuth as add_panel_azimuth_test,
    create_datetime_for_test_data,
    merge_solar_data as merge_solar_data_test,
    remove_datetime_column as remove_datetime_column_test,
)
from feature_engineering import perform_feature_engineering
from scaling import scale_data
from model_training import train_and_evaluate_model, predict_test_data, train_stacking_model
from submission import create_submission_file
from sklearn.model_selection import train_test_split


def main():
    # 讀取資料
    train_data, test_data, solar_angles, solar_azimuths = load_data()
    print("資料已成功讀取。")

    # 處理資料
    solar_angles = process_solar_angles(solar_angles)
    train_data = create_datetime_for_train_data(train_data)
    solar_azimuths = process_solar_azimuths(solar_azimuths)
    train_data = merge_solar_data(train_data, solar_angles, solar_azimuths)
    train_data = add_panel_azimuth(train_data)
    print("已將太陽能相關資料合併到 train_data。")
    train_data = drop_unnecessary_columns(train_data)
    print("已從 train_data 中移除不必要的欄位。")
    train_data = remove_datetime_column_train(train_data)

    # 處理測試資料
    test_data_features = convert_test_data_index(test_data)
    test_data_features = add_panel_azimuth_test(test_data_features)
    print("已將序號轉換為特徵。")
    test_data_features = create_datetime_for_test_data(test_data_features)
    test_data_features = merge_solar_data_test(test_data_features, solar_angles, solar_azimuths)
    print("已將太陽能相關資料合併到 test_data_features。")
    test_data_features = remove_datetime_column_test(test_data_features)
    print("已從 test_data_features 中移除 Datetime 欄位。")

    # 準備訓練和驗證集
    X = train_data.drop(columns=["Power"])
    y = train_data["Power"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("已將資料切分為訓練集和驗證集。")

    # 特徵工程
    use_openfe = True
    if use_openfe:
        X_train_fe, X_val_fe, test_data_features_fe = perform_feature_engineering(
            X_train, y_train, X_val, test_data_features, n_features=5
        )
    else:
        X_train_fe = X_train.copy()
        X_val_fe = X_val.copy()
        test_data_features_fe = test_data_features.copy()

    print("特徵工程已完成。")

    # 資料縮放
    X_train_fe, X_val_fe, test_data_features_fe = scale_data(
        X_train_fe, X_val_fe, test_data_features_fe
    )
    print("資料縮放已完成。")

    # 訓練和評估模型
    use_optuna = False  # 將此設為 True 以啟用 Optuna 調參
    best_params_xgb, best_params_lgb, best_params_cat = train_and_evaluate_model(
        X_train_fe, y_train, X_val_fe, y_val, use_optuna=use_optuna
    )

    # 訓練模型與預測測試資料
    y_test_pred = predict_test_data(
        X_train_fe,
        y_train,
        X_val_fe,
        y_val,
        test_data_features_fe,
        best_params_xgb,
        best_params_lgb,
        best_params_cat,
    )

    # 建立提交檔案
    create_submission_file(test_data, y_test_pred)
    print("結果已儲存至 submission.csv")

    # 輸出最佳參數
    print("最佳參數（XGBoost）：", best_params_xgb)
    print("最佳參數（LightGBM）：", best_params_lgb)
    print("最佳參數（CatBoost）：", best_params_cat)


if __name__ == "__main__":
    main()
