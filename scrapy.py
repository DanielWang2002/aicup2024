import time
import csv
from selenium.webdriver.firefox.options import Options
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from webdriver_manager.firefox import GeckoDriverManager


def extract_data(driver):
    """
    從當前頁面中擷取日期與天文相關資料，包括日出時刻、太陽過中天、日沒時刻、仰角、方位角等。
    """
    try:
        # 擷取日期
        date_text = driver.find_element(By.CSS_SELECTOR, "h4.d-title").text
        date = date_text.split(" ")[-1].strip("()")

        # 定位父元素
        parent_div = driver.find_element(By.XPATH, "//div[@class='flex_table sunrise']")

        # 在父元素內定位所有表格行
        rows = parent_div.find_elements(By.XPATH, ".//tr")
        table_data = {}
        angle_id = 0

        # 遍歷每行，提取 <th> 和 <td> 值
        for row in rows:
            th_element = row.find_element(By.TAG_NAME, "th").text  # 標題
            td_element = row.find_element(By.TAG_NAME, "td").text  # 值

            if th_element == '方位角':
                # 因為有兩個方位角欄位，使用 angle_id 來區分
                table_data[th_element + str(angle_id)] = td_element
                angle_id += 1
            else:
                table_data[th_element] = td_element

        return date, table_data
    except Exception as e:
        print("資料擷取失敗：", e)
        return None, None


if __name__ == "__main__":
    # 設定 WebDriver，使用 webdriver_manager 自動安裝 ChromeDriver
    options = Options()

    # 使用 GeckoDriverManager 自動下載 geckodriver，並使用 Firefox 瀏覽器
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)

    # 開啟目標網頁
    url = "https://www.cwa.gov.tw/V8/C/K/astronomy_day.html"
    driver.get(url)

    # 等待頁面加載
    time.sleep(2)

    # 選擇「花蓮縣」
    select_element = driver.find_element(By.ID, "area")
    select = Select(select_element)
    select.select_by_visible_text("花蓮縣")

    # 等待頁面更新
    time.sleep(2)

    # 找到上一日按鈕
    back_button = driver.find_element(By.ID, "back")

    # 儲存結果的字典
    results = {}

    # 爬取兩年的資料（730 天）
    for _ in range(730):  # 365 * 2 天
        # 提取資料
        date, data = extract_data(driver)
        if date and data:
            results[date] = data
            print(f"爬取日期: {date}, 資料: {data}")

        # 按下上一日按鈕
        back_button.click()

        # 等待頁面更新
        time.sleep(2)

    # 關閉瀏覽器
    driver.quit()

    # 將結果保存為 CSV
    csv_file = "astronomy_data.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # 寫入表頭
        writer.writerow(
            ["日期", "日出時刻", "日出方位角", "太陽過中天", "仰角", "日沒時刻", "日沒方位角"]
        )

        # 寫入資料
        for date, values in results.items():
            writer.writerow(
                [
                    date,
                    values["日出時刻"],
                    values["方位角0"],
                    values["太陽過中天"],
                    values["仰角"],
                    values["日沒時刻"],
                    values["方位角1"],
                ]
            )

    print(f"資料已保存至 {csv_file}")
