import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import List, Dict


def calculate_elevation(observation_time: str, data_row: pd.Series) -> float:
    """
    計算某時間點的太陽仰角。

    Args:
        observation_time (str): 時間點 (格式: HH:MM)。
        data_row (pd.Series): 包含日出、日落、太陽過中天等資料的行數據。

    Returns:
        float: 該時間點的太陽仰角。
    """
    obs_time = datetime.strptime(observation_time, "%H:%M").time()
    current_hour = obs_time.hour + obs_time.minute / 60

    # 提取日出、日落、過中天時間
    sunrise = datetime.strptime(data_row["日出時刻"], "%H:%M").time()
    sunset = datetime.strptime(data_row["日沒時刻"], "%H:%M").time()
    solar_noon = datetime.strptime(data_row["太陽過中天"], "%H:%M").time()

    sunrise_hour = sunrise.hour + sunrise.minute / 60
    sunset_hour = sunset.hour + sunset.minute / 60
    noon_hour = solar_noon.hour + solar_noon.minute / 60

    # 最大仰角處理
    max_elevation = float(data_row["仰角"].replace("S", ""))

    # 在日出前或日落後，直接返回 0
    if current_hour < sunrise_hour or current_hour > sunset_hour:
        return 0.0

    # 計算仰角
    time_fraction = (current_hour - noon_hour) / (sunset_hour - sunrise_hour) * np.pi
    elevation_angle = max_elevation * np.cos(time_fraction)
    return max(0.0, elevation_angle)


def generate_time_series(start: str = "05:00", end: str = "19:00", interval: int = 10) -> List[str]:
    """
    生成時間序列，每個時間間隔為 10 分鐘。

    Args:
        start (str): 起始時間 (格式: HH:MM)，默認為 05:00。
        end (str): 結束時間 (格式: HH:MM)，默認為 19:00。
        interval (int): 時間間隔（分鐘），默認為 10。

    Returns:
        List[str]: 時間序列列表，格式為 HH:MM。
    """
    start_time = datetime.strptime(start, "%H:%M")
    end_time = datetime.strptime(end, "%H:%M")
    times = []

    while start_time <= end_time:
        times.append(start_time.strftime("%H:%M"))
        start_time += timedelta(minutes=interval)

    return times


def process_daily_data(times: List[str], daily_row: pd.Series) -> List[Dict[str, object]]:
    """
    處理單天的太陽仰角數據。

    Args:
        times (List[str]): 每天的時間序列。
        daily_row (pd.Series): 包含當天的日出、日落和仰角數據。

    Returns:
        List[Dict[str, object]]: 當天每個時間點的仰角結果。
    """
    daily_results = []
    for time_str in times:
        angle = calculate_elevation(time_str, daily_row)
        daily_results.append({"日期": daily_row["日期"], "時間": time_str, "仰角": round(angle, 2)})
    return daily_results


def main(input_file: str, output_file: str) -> None:
    """
    計算太陽仰角並保存到 CSV。

    Args:
        input_file (str): 輸入的 CSV 文件路徑。
        output_file (str): 輸出的 CSV 文件路徑。
    """
    data = pd.read_csv(input_file)
    time_series = generate_time_series()

    all_results = []
    for _, row in data.iterrows():
        all_results.extend(process_daily_data(time_series, row))

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(result_df.head())


if __name__ == "__main__":
    main("astronomy_data.csv", "solar_angles_10min.csv")
