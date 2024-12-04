# data_loader.py

import pandas as pd


def load_data() -> tuple:
    """
    讀取訓練資料、測試資料和太陽能相關資料。

    :return: 包含 train_data, test_data, solar_angles, solar_azimuths 的元組
    """
    train_data = pd.read_csv("./concat.csv")
    test_data = pd.read_csv("./aicup_empty_answer.csv")
    solar_angles = pd.read_csv("./solar_angles_10min.csv")
    solar_azimuths = pd.read_csv("./solar_azimuths.csv")

    return train_data, test_data, solar_angles, solar_azimuths


def load_location_azimuths() -> dict:
    """
    讀取 location_azimuths.csv 並轉換為字典。

    :return: dict，LocationCode 對應的太陽能板方位
    """
    df = pd.read_csv('location_azimuths.csv')
    location_azimuths_dict = dict(zip(df['LocationCode'], df['太陽能板方位']))
    return location_azimuths_dict
