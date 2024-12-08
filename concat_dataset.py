import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.DataFrame()

listdir = os.listdir('./dataset/aicup')
listdir.sort()

for file in listdir:
    # 匯入L1_Train.csv, L2_Train.csv, L3_Train.csv, ... L17_Train.csv
    if (
        file.endswith('.csv')
        and file.startswith('L')
        and 'predicted' not in file
        and 'cleaned' not in file
    ):
        df = pd.concat([df, pd.read_csv(f'./dataset/aicup/{file}')], ignore_index=True)
        # L2, L4, L7, L8, L9, L10, L12
        if file in [
            'L2_Train.csv',
            'L4_Train.csv',
            'L7_Train.csv',
            'L8_Train.csv',
            'L9_Train.csv',
            'L10_Train.csv',
            'L12_Train.csv',
        ]:
            add_file = file.replace('Train', 'Train_2')
            additional_df = pd.read_csv(f'./dataset/aicup_2/{add_file}')
            df = pd.concat([df, additional_df], ignore_index=True)

df.columns = df.columns.str.replace(r"\(.*\)", "", regex=True)

# 確保 'DateTime' 欄位是 datetime 格式
df['DateTime'] = pd.to_datetime(df['DateTime'])

# 過濾掉相鄰時間差不是 1 分鐘的資料
df = df[df['DateTime'].diff().dt.total_seconds() == 60]

# 將 'DateTime' 的分鐘部分取整到最近的 10 分鐘
df['DateTime'] = df['DateTime'].dt.floor('10min')

# 按照 'DateTime' 和 'LocationCode' 進行聚合
df = df.groupby(['DateTime', 'LocationCode']).mean().reset_index()

print(df.shape)
print(df.head())

df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Hour'] = df['DateTime'].dt.hour
df['Minute'] = df['DateTime'].dt.minute
df = df.drop(columns=['DateTime'])
print(df.head())
print(df.shape)

df.to_csv('./concat.csv', index=False)
