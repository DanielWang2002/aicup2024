# data_preprocessing.py

import pandas as pd
from data_loader import load_location_azimuths


def process_solar_angles(solar_angles: pd.DataFrame) -> pd.DataFrame:
    """
    處理 solar_angles 資料，將「日期」和「時間」合併成 Datetime 格式。

    :param solar_angles: solar_angles 資料的 DataFrame
    :return: 處理後的 solar_angles DataFrame
    """
    solar_angles["Datetime"] = pd.to_datetime(
        solar_angles["日期"].astype(str) + " " + solar_angles["時間"].astype(str)
    )
    solar_angles = solar_angles.drop(columns=["日期", "時間"])
    return solar_angles


def create_datetime_for_train_data(train_data: pd.DataFrame) -> pd.DataFrame:
    """
    處理 train_data 資料，將「Month」、「Day」、「Hour」、「Minute」合併成 Datetime 格式。

    :param train_data: 訓練資料的 DataFrame
    :return: 添加 Datetime 欄位後的 train_data DataFrame
    """
    train_data["Datetime"] = pd.to_datetime(
        "2024"
        + train_data["Month"].astype(str).str.zfill(2)
        + train_data["Day"].astype(str).str.zfill(2)
        + " "
        + train_data["Hour"].astype(str).str.zfill(2)
        + ":"
        + train_data["Minute"].astype(str)
    )
    return train_data


def process_solar_azimuths(solar_azimuths: pd.DataFrame) -> pd.DataFrame:
    """
    將 solar_azimuths 的 DateTime 轉換成 Datetime 格式。

    :param solar_azimuths: solar_azimuths 資料的 DataFrame
    :return: 處理後的 solar_azimuths DataFrame
    """
    solar_azimuths["Datetime"] = pd.to_datetime(solar_azimuths["DateTime"])
    solar_azimuths = solar_azimuths.drop(columns=["DateTime"])
    return solar_azimuths


def merge_solar_data(
    train_data: pd.DataFrame, solar_angles: pd.DataFrame, solar_azimuths: pd.DataFrame
) -> pd.DataFrame:
    """
    合併 solar_angles 和 solar_azimuths 到 train_data。

    :param train_data: 訓練資料的 DataFrame
    :param solar_angles: 處理後的 solar_angles DataFrame
    :param solar_azimuths: 處理後的 solar_azimuths DataFrame
    :return: 合併後的 train_data DataFrame
    """
    train_data = train_data.merge(solar_angles, on="Datetime", how="left")
    train_data = train_data.merge(solar_azimuths, on="Datetime", how="left")
    return train_data


def add_panel_azimuth(train_data: pd.DataFrame) -> pd.DataFrame:
    """
    新增「太陽能板方位」欄位到 train_data。

    :param train_data: 訓練資料的 DataFrame
    :return: 添加「太陽能板方位」欄位後的 train_data DataFrame
    """
    location_azimuths_dict = load_location_azimuths()
    train_data["太陽能板方位"] = train_data["LocationCode"].map(location_azimuths_dict)
    return train_data


def drop_unnecessary_columns(train_data: pd.DataFrame) -> pd.DataFrame:
    """
    移除不需要的欄位。

    :param train_data: 訓練資料的 DataFrame
    :return: 移除不必要欄位後的 train_data DataFrame
    """
    train_data = train_data.drop(
        columns=["WindSpeed", "Pressure", "Temperature", "Humidity", "Sunlight"]
    )
    return train_data


def remove_datetime_column(train_data: pd.DataFrame) -> pd.DataFrame:
    """
    移除 Datetime 欄位。

    :param train_data: 訓練資料的 DataFrame
    :return: 移除 Datetime 欄位後的 train_data DataFrame
    """
    train_data = train_data.drop(columns=["Datetime"])
    return train_data
