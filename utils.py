# utils.py

import pandas as pd


def parse_index(index: str) -> pd.Series:
    """
    解析序號，將序號轉換成特徵。

    :param index: 序號字串
    :return: 包含 Month, Day, Hour, Minute, LocationCode 的 Series
    """
    index = str(index)
    date = index[:8]
    time = index[8:12]
    location_code = index[12:]

    # 提取年月日
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])

    # 提取時分
    hour = int(time[:2])
    minute = int(time[2:])

    # 提取 LocationCode，去掉前導 0
    location_code = int(location_code.lstrip("0")) if location_code.lstrip("0") else 0

    return pd.Series(
        {
            "Month": month,
            "Day": day,
            "Hour": hour,
            "Minute": minute,
            "LocationCode": location_code,
        }
    )
