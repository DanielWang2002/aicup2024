# test_data_preprocessing.py

import pandas as pd

from data_loader import load_location_azimuths
from utils import parse_index


def convert_test_data_index(test_data: pd.DataFrame) -> pd.DataFrame:
    """
    將測試資料的序號轉換為特徵。

    :param test_data: 測試資料的 DataFrame
    :return: 包含特徵的 test_data_features DataFrame
    """
    test_data_features = test_data["序號"].apply(parse_index)
    return test_data_features


def add_panel_azimuth(test_data_features: pd.DataFrame) -> pd.DataFrame:
    """
    新增「太陽能板方位」欄位到 test_data_features。

    :param test_data_features: 測試資料特徵的 DataFrame
    :return: 添加「太陽能板方位」欄位後的 test_data_features DataFrame
    """
    location_azimuths_dict = load_location_azimuths()
    test_data_features["太陽能板方位"] = test_data_features["LocationCode"].map(
        location_azimuths_dict
    )
    return test_data_features


def create_datetime_for_test_data(test_data_features: pd.DataFrame) -> pd.DataFrame:
    """
    建立 Datetime 欄位給 test_data_features。

    :param test_data_features: 測試資料特徵的 DataFrame
    :return: 添加 Datetime 欄位後的 test_data_features DataFrame
    """
    test_data_features["Datetime"] = pd.to_datetime(
        "2024"
        + test_data_features["Month"].astype(str).str.zfill(2)
        + test_data_features["Day"].astype(str).str.zfill(2)
        + " "
        + test_data_features["Hour"].astype(str).str.zfill(2)
        + ":"
        + test_data_features["Minute"].astype(str)
    )
    return test_data_features


def merge_solar_data(
    test_data_features: pd.DataFrame, solar_angles: pd.DataFrame, solar_azimuths: pd.DataFrame
) -> pd.DataFrame:
    """
    合併 solar_angles 和 solar_azimuths 到 test_data_features。

    :param test_data_features: 測試資料特徵的 DataFrame
    :param solar_angles: 處理後的 solar_angles DataFrame
    :param solar_azimuths: 處理後的 solar_azimuths DataFrame
    :return: 合併後的 test_data_features DataFrame
    """
    test_data_features = test_data_features.merge(solar_angles, on="Datetime", how="left")
    test_data_features = test_data_features.merge(solar_azimuths, on="Datetime", how="left")
    return test_data_features


def remove_datetime_column(test_data_features: pd.DataFrame) -> pd.DataFrame:
    """
    移除 Datetime 欄位。

    :param test_data_features: 測試資料特徵的 DataFrame
    :return: 移除 Datetime 欄位後的 test_data_features DataFrame
    """
    test_data_features = test_data_features.drop(columns=["Datetime"])
    return test_data_features
