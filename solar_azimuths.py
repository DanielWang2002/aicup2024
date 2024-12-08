import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List


def compute_azimuth_series(
    date: str, sunrise: str, sunset: str, sunrise_az: float, sunset_az: float
) -> List[float]:
    """
    根據日出和日沒時間及方位角計算每10分鐘的太陽方位角序列。

    Args:
        date (str): 日期 (格式: YYYY/MM/DD)。
        sunrise (str): 日出時間 (格式: HH:MM)。
        sunset (str): 日沒時間 (格式: HH:MM)。
        sunrise_az (float): 日出方位角。
        sunset_az (float): 日沒方位角。

    Returns:
        List[float]: 每10分鐘的太陽方位角序列。
    """
    sunrise_dt = datetime.strptime(sunrise, "%H:%M")
    sunset_dt = datetime.strptime(sunset, "%H:%M")
    total_day_minutes = (sunset_dt - sunrise_dt).seconds / 60

    delta_azimuth = sunset_az - sunrise_az
    times_in_minutes = np.arange(0, total_day_minutes + 10, 10)

    azimuths = [
        sunrise_az + delta_azimuth * (1 - np.cos(np.pi * (t / total_day_minutes))) / 2
        for t in times_in_minutes
    ]
    return azimuths


def generate_azimuth_dataframe(row: pd.Series) -> pd.DataFrame:
    """
    生成特定日期的太陽方位角資料表。

    Args:
        row (pd.Series): 包含日期、日出時間、日沒時間及方位角的資料。

    Returns:
        pd.DataFrame: 包含日期時間及對應方位角的資料表。
    """
    date = datetime.strptime(row["日期"], "%Y/%m/%d")
    sunrise = row["日出時刻"]
    sunset = row["日沒時刻"]
    sunrise_az = float(row["日出方位角"])
    sunset_az = float(row["日沒方位角"])

    azimuths = compute_azimuth_series(
        date=str(date.date()),
        sunrise=sunrise,
        sunset=sunset,
        sunrise_az=sunrise_az,
        sunset_az=sunset_az,
    )

    result = []
    current_time = datetime.combine(date, datetime.strptime("05:00", "%H:%M").time())
    end_time = datetime.combine(date, datetime.strptime("19:00", "%H:%M").time())

    while current_time <= end_time:
        if current_time < datetime.combine(
            date, datetime.strptime(sunrise, "%H:%M").time()
        ) or current_time > datetime.combine(date, datetime.strptime(sunset, "%H:%M").time()):
            result.append((current_time.strftime("%Y/%m/%d %H:%M"), 0.0))
        else:
            elapsed_minutes = (
                current_time - datetime.combine(date, datetime.strptime(sunrise, "%H:%M").time())
            ).seconds / 60
            index = int(elapsed_minutes / 10)
            azimuth_value = azimuths[index] if index < len(azimuths) else 0.0
            result.append((current_time.strftime("%Y/%m/%d %H:%M"), round(azimuth_value, 2)))

        current_time += timedelta(minutes=10)

    return pd.DataFrame(result, columns=["DateTime", "方位角"])


def main() -> None:
    """
    處理原始數據並輸出完整的太陽方位角資料。
    """
    input_file = "./astronomy_data.csv"
    output_file = "./solar_azimuths.csv"

    data = pd.read_csv(input_file)
    all_azimuths = [generate_azimuth_dataframe(row) for _, row in data.iterrows()]

    final_result = pd.concat(all_azimuths, ignore_index=True)
    final_result.to_csv(output_file, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
