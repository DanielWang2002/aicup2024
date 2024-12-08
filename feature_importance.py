import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False


def plot_feature_importance(stack_model, feature_columns, output_path="feature_importance.png"):
    """
    繪製特徵重要度圖並將圖片儲存。

    :param stack_model: 已訓練完成的堆疊模型 (StackingRegressor)
    :param feature_columns: 特徵欄位名稱列表
    :param output_path: 輸出圖片的檔案路徑，預設為 "feature_importance.png"
    """
    # 從堆疊模型中取得基學習器
    estimators_dict = stack_model.named_estimators_

    # 建立一個 DataFrame 來儲存各模型的特徵重要度
    importance_df = pd.DataFrame(index=feature_columns)

    # 取得 XGBoost 特徵重要度
    if "xgb" in estimators_dict:
        xgb_model = estimators_dict["xgb"]
        if hasattr(xgb_model, "feature_importances_"):
            importance_df["XGBoost"] = xgb_model.feature_importances_

    # 取得 LightGBM 特徵重要度
    if "lgb" in estimators_dict:
        lgb_model = estimators_dict["lgb"]
        if hasattr(lgb_model, "feature_importances_"):
            importance_df["LightGBM"] = lgb_model.feature_importances_

    # 取得 CatBoost 特徵重要度
    if "cat" in estimators_dict:
        cat_model = estimators_dict["cat"]
        if hasattr(cat_model, "feature_importances_"):
            importance_df["CatBoost"] = cat_model.feature_importances_

    # 將無法取得特徵重要度的模型忽略
    importance_df = importance_df.dropna(axis=1, how="all")

    # 若沒有任何特徵重要度可用，則直接返回
    if importance_df.empty:
        print("無法取得特徵重要度資訊。")
        return

    # 計算平均特徵重要度作為最終排序依據
    importance_df["Mean"] = importance_df.mean(axis=1)

    # 根據平均重要度排序
    importance_df = importance_df.sort_values(by="Mean", ascending=False)

    # 繪製長條圖
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df.index, importance_df["Mean"], color="skyblue")
    plt.gca().invert_yaxis()  # 使重要度最高的特徵排在最上方
    plt.title("特徵重要度 (平均值)")
    plt.xlabel("重要度")
    plt.ylabel("特徵名稱")
    plt.tight_layout()

    # 儲存圖片
    plt.savefig(output_path)
    plt.close()
    print(f"特徵重要度圖已儲存至 {output_path}")
