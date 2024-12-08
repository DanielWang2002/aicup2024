import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the uploaded dataset
file_path = '../concat.csv'
data = pd.read_csv(file_path)
# 地區分類
data['Class'] = data['LocationCode'].apply(lambda i: 1 if i <= 14 else (2 if i <= 16 else 3))
data = data.sort_values(by=['Month', 'Day', 'Hour', 'Minute']).reset_index(drop=True)
print(len(data))

# 轉換為一天中的分鐘數
data['MinutesOfDay'] = data['Hour'] * 60 + data['Minute']

# 週期性編碼
data['Time_sin'] = np.sin(2 * np.pi * data['MinutesOfDay'] / (24 * 60))
data['Time_cos'] = np.cos(2 * np.pi * data['MinutesOfDay'] / (24 * 60))

# Step 1: Convert all data to float and remove rows with non-convertible values
for column in data.columns:
    if column not in ['Month', 'Day', 'Hour', 'Minute', 'Class']:
        try:
            data[column] = data[column].astype(float)
        except ValueError:
            data = data.drop(columns=[column])

# Remove rows with NaN values created during the conversion
data = data.dropna()

scaler1 = StandardScaler()
data['Sunlight(Lux)_normalized'] = scaler1.fit_transform(data[['Sunlight']])

solar_path = './solar_angles_10min_new.csv'
solar_data = pd.read_csv(solar_path)
solar_data_unique = solar_data.drop_duplicates(subset=['Month', 'Day', 'Hour', 'Minute'])

# 左合併，確保原資料長度不變
data = pd.merge(data, solar_data_unique, on=['Month', 'Day', 'Hour', 'Minute'], how='left')

# Step 2: Calculate correlations with the target
target = 'Power'
excluded_columns = ["AvgTemp"]
remaining_columns = [col for col in data.columns if col not in excluded_columns]
data = data[remaining_columns]
correlation = data.corr()

# Find features with a correlation > 0.3 with "Power(mW)"
features_high_corr = correlation[(abs(correlation["Power"]) > 0.4)].index

# Exclude the target itself from the features
features_high_corr = [feat for feat in features_high_corr]
features_high_corr.append('Class')

# Display the selected features
print(features_high_corr)

print("與(Power(mW))的相關性：")
print(correlation[target].sort_values(ascending=False))

# features_high_corr = [feat for feat in features_high_corr if feat not in ['Sunlight(Lux)', 'Hour', 'feature_Temperature', 'Class', target]]
features_high_corr = ['Hour', 'Sunlight(Lux)_normalized', '仰角', 'Class', 'Power', 'Time_sin']
print(features_high_corr)

df_selected = data[features_high_corr]
df_selected.dropna(inplace=True)
df_selected.reset_index(drop=True)
print(df_selected)

# Split dataset based on the 'Class' column into separate groups
class_datasets = {
    class_label: group.drop(columns=["Class"])
    for class_label, group in df_selected.groupby("Class")
}

features_high_corr = ['Sunlight(Lux)_normalized', '仰角', 'Time_sin']

from sklearn.model_selection import train_test_split

# Split each class dataset into train and test sets (8:2 ratio)
split_data = {
    class_label: train_test_split(data, test_size=0.2, random_state=42, shuffle=False)
    for class_label, data in class_datasets.items()
}

from torch.utils.data import DataLoader, TensorDataset


# Prepare datasets for training and testing
def prepare_dataloader(data, batch_size=128):
    feature_columns = features_high_corr
    X = (
        torch.tensor(data[feature_columns].values, dtype=torch.float32).unsqueeze(1).to(device)
    )  # Add sequence dimension
    y = torch.tensor(data[target].values, dtype=torch.float32).to(device)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# 定義 Attention 層
class Attention(nn.Module):
    def __init__(self, hidden_layer_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_layer_size, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, time_steps, hidden_layer_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: (batch_size, time_steps, 1)
        weighted_output = (
            lstm_output * attention_weights
        )  # (batch_size, time_steps, hidden_layer_size)
        output = torch.sum(weighted_output, dim=1)  # 加權後的輸出 (batch_size, hidden_layer_size)
        return output


# LSTM Model
class LSTM_MLP_Model(nn.Module):
    def __init__(
        self,
        input_size=5,
        hidden_layer_size=256,
        output_size=1,
        mlp_hidden_size_1=256,
        mlp_hidden_size_2=256,
    ):
        super(LSTM_MLP_Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=12, batch_first=True)
        self.hidden_layer_size = hidden_layer_size
        self.attention = Attention(self.hidden_layer_size)
        # self.softmax = nn.Softmax()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_layer_size, mlp_hidden_size_1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size_1, mlp_hidden_size_2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size_2, output_size),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM 輸出
        # lstm_out = lstm_out[:, -1, :]  # 只取最後一個時間步的輸出
        attention_out = self.attention(lstm_out)
        predictions = self.mlp(attention_out)  # 輸入到 MLP
        return predictions, lstm_out[:, -1, :]  # 同時返回 MLP 的最終預測和 LSTM 的特徵


# Initialize models, dataloaders, and training configurations
models = {}
dataloaders = {}
hidden_layer_size = 256
mlp_hidden_size_1 = 512
mlp_hidden_size_2 = 1024
learning_rate = 0.001
output_size = 1


# Training function
def train_model(model, xgb_model, train_loader, criterion, optimizer, num_epochs=1000):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs, lstm_out = model(inputs)
            # xgb 訓練資料
            lstm_train_features_np = lstm_out.cpu().detach().numpy()
            targets_np = targets.cpu().detach().numpy()
            # XGBoost 模型訓練
            xgb_model.fit(lstm_train_features_np, targets_np)
            # XGBoost 模型預測
            y_pred_xgb = xgb_model.predict(lstm_train_features_np)
            y_pred_xgb_tensor = torch.tensor(y_pred_xgb, dtype=torch.float32).to(device)
            loss = criterion(outputs.squeeze(), targets)
            # 合併損失值
            loss_lstm = criterion(outputs.squeeze(), targets)
            loss_xgb = criterion(y_pred_xgb_tensor, targets)
            total_loss = (loss_lstm + loss_xgb) / 2
            total_loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")


# Testing function
def test_model(model, xgb_model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs, lstm_out = model(inputs)
            # xgb 訓練資料
            lstm_train_features_np = lstm_out.cpu().detach().numpy()
            targets_np = targets.cpu().detach().numpy()
            # XGBoost 模型訓練
            xgb_model.fit(lstm_train_features_np, targets_np)
            # XGBoost 模型預測
            y_pred_xgb = xgb_model.predict(lstm_train_features_np)
            y_pred_xgb_tensor = torch.tensor(y_pred_xgb, dtype=torch.float32).to(device)
            loss = criterion(outputs.squeeze(), targets)
            # 合併損失值
            loss_lstm = criterion(outputs.squeeze(), targets)
            loss_xgb = criterion(y_pred_xgb_tensor, targets)
            total_loss = (loss_lstm + loss_xgb) / 2
            total_loss += total_loss
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


models = {}
dataloaders = {}
num_epochs = 600

xgb_models = {}

for class_label, (train_data, test_data) in split_data.items():
    input_size = len(features_high_corr)

    # Initialize the model and move it to the correct device (GPU or CPU)
    models[class_label] = LSTM_MLP_Model(
        input_size=input_size,
        hidden_layer_size=hidden_layer_size,
        mlp_hidden_size_1=mlp_hidden_size_1,
        mlp_hidden_size_2=mlp_hidden_size_2,
        output_size=output_size,
    ).to(device)
    # 初始化 XGBoost 模型
    xgb_models[class_label] = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=20,
        max_depth=8,
        learning_rate=0.01,
        # tree_method="gpu_hist",  # 使用 GPU 加速
        # predictor="gpu_predictor"  # 使用 GPU 進行預測
    )

    # Prepare data loaders for training and testing
    dataloaders[class_label] = {
        "train": prepare_dataloader(train_data, batch_size=1024),
        "test": prepare_dataloader(test_data, batch_size=128),
    }

    # Define the loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(models[class_label].parameters(), lr=learning_rate)

    # Train the model
    print(f"Training model for Class {class_label}...")
    train_model(
        models[class_label],
        xgb_models[class_label],
        dataloaders[class_label]["train"],
        criterion,
        optimizer,
        num_epochs,
    )

    # Test the model
    print(f"Testing model for Class {class_label}...")
    test_loss = test_model(
        models[class_label], xgb_models[class_label], dataloaders[class_label]["test"], criterion
    )

    print(f"Class {class_label} Test Loss: {test_loss:.4f}")

# 欄位名與目標欄位
feature_columns = features_high_corr

# 讀取檔案
file_path = '../比賽資料/36_TestSet_SubmissionTemplate/match_merge1.csv'
df = pd.read_csv(file_path)


# 預測發電量
predictions = []

for class_label, model in models.items():
    # 選擇當前類別的數據
    class_data = df[df["Class"] == class_label]
    if class_data.empty:
        continue  # 如果該類別無數據，跳過

    # 提取特徵
    features = class_data[feature_columns]

    # 將特徵轉換為 PyTorch 張量
    features_tensor = torch.tensor(features.values, dtype=torch.float32).to(device)

    # 模型預測
    model.eval()
    with torch.no_grad():
        outputs, lstm_out = model(features_tensor.unsqueeze(1))  # 添加序列維度

        # xgb 訓練資料
        lstm_val_features_np = lstm_out.cpu().detach().numpy()
        y_pred_xgb = xgb_models[class_label].predict(lstm_val_features_np)

        lstm_outputs = outputs.squeeze().cpu().numpy()  # LSTM 預測值
        final_predictions = (lstm_outputs * 0.3 + y_pred_xgb * 0.7) / 2  # 取平均

        # 儲存到 DataFrame
        class_data["答案"] = final_predictions

    # 紀錄結果
    predictions.append(class_data[["序號", "答案"]])

# 合併所有類別的預測結果
result = pd.concat(predictions)

# 儲存預測結果
output_path = '../比賽資料/36_TestSet_SubmissionTemplate/up_lstm_xgb.csv'  # 替換為實際輸出檔案路徑
result.to_csv(output_path, index=False)

print(f"預測完成，結果已儲存到 {output_path}")


# 檢查每個欄位中缺失值的數量
missing_columns = result.isnull().sum()

# 篩選出有缺失值的欄位
missing_columns = missing_columns[missing_columns > 0]

# 顯示有缺失值的欄位及其對應的缺失數量
print("欄位及缺失值數量：")
print(missing_columns)

# 找出有缺失值的行的索引
print("\n包含缺失值的索引：")
missing_rows = result[result.isnull().any(axis=1)]
print((missing_rows.index))
