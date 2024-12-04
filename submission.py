# submission.py

import pandas as pd


def create_submission_file(
    test_data: pd.DataFrame, y_test_pred: pd.Series, output_path: str = "./submission.csv"
):
    """
    建立提交檔案。

    :param test_data: 測試資料的 DataFrame
    :param y_test_pred: 測試資料的預測結果 Series
    :param output_path: 輸出檔案的路徑，預設為 "./submission.csv"
    """
    test_data["答案"] = y_test_pred
    test_data[["序號", "答案"]].to_csv(output_path, index=False)
