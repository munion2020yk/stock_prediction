import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import random  # random 모듈 추가

# 폰트 설정 (코랩/영문 환경 호환)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False 

# --- 1. Configuration ---
CONFIG = {
    "data_file": "KOSPI_dataset_final.csv",
    "data_start": "2013-08-06",
    "data_end": "2025-11-28", 
    
    # 검증 대상 기간
    "test_start_date": "2025-11-24",
    "test_end_date": "2025-11-28",
    
    "seq_length": 5,          
    "predict_horizon": 5,     
    
    "hidden_size": 256,
    "num_layers": 1,
    "num_classes": 1,         
    
    "cnn_num_layers": 1,
    "num_filters": 32,
    "kernel_size": 5,
    
    "batch_size": 256,
    "epochs": 100,
    "learning_rate": 0.005,
    "patience": 5,
    
    "seed": 100, # [추가] 랜덤 시드 설정
    
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- Feature Groups ---
FEATURE_GROUPS = {
    "KOSPI": ['KOSPI_Close', 'KOSPI_Open', 'KOSPI_High', 'KOSPI_Low', 'KOSPI_Volume', 'KOSPI_Amount', 'KOSPI_Change', 'KOSPI_Fluctuation', 'KOSPI_UpDown'],
    "NASDAQ": ['NAS_Open', 'NAS_High', 'NAS_Low', 'NAS_Close', 'NAS_Volume', 'NAS_Change'],
    "VKOSPI": ['VKOSPI_Close', 'VKOSPI_Change'],
    "Rate_FX": ['USD_KRW', 'EUR_KRW', 'Rate'],
    "Foreign": ['Foreign_MarketCap_Ratio', 'Foreign_MarketCap', 'Foreign_Rate'],
    "Future": ['Future_Close', 'Future_Change'],
    "Oil": ['WTI_Close', 'WTI_Change']
}

# LSTM+ 모델용 특정 피처 리스트
LSTM_PLUS_FEATURES = [
    'KOSPI_Open', 'KOSPI_High', 'KOSPI_Low', 'KOSPI_Close', 'KOSPI_Volume', 
    'Future_Close', 'Future_Change', 
    'USD_KRW', 
    'WTI_Change'
]

print(f"Using Device: {CONFIG['device']}")

# --- [추가] 랜덤 시드 고정 함수 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시
        # CuDNN 결정론적 모드 설정 (속도는 느려질 수 있음)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")

# --- 2. Data Processing Utils ---
def load_data(config):
    if not os.path.exists(config["data_file"]):
        raise FileNotFoundError(f"File not found: {config['data_file']}")
    
    encodings = ['utf-16', 'utf-8', 'utf-8-sig', 'cp949', 'latin1']
    df = None
    for enc in encodings:
        try:
            temp_df = pd.read_csv(config["data_file"], sep='\t', index_col="Date", parse_dates=True, encoding=enc)
            if len(temp_df.columns) > 1: df = temp_df; break
            temp_df = pd.read_csv(config["data_file"], sep=',', index_col="Date", parse_dates=True, encoding=enc)
            if len(temp_df.columns) > 1: df = temp_df; break
        except: continue
            
    if df is None: raise ValueError("Failed to read file.")
    
    for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.loc[config["data_start"]:config["data_end"]]
    df = df.ffill().bfill()
    df.dropna(inplace=True)
    df.columns = [c.strip() for c in df.columns]
    return df

def process_features_and_split(df, drop_group_name=None, select_features_list=None):
    target_col = "KOSPI_Close"
    available_cols = df.columns.tolist()
    
    # 1. Target (y)
    raw_y = df[[target_col]].values
    
    # 2. Input (X) 구성
    if select_features_list:
        # [LSTM+용] Whitelist 방식
        selected_cols = [c for c in select_features_list if c in available_cols]
        input_df = df[selected_cols]
    else:
        # [기존 모델용] Blacklist 방식 (Drop Group)
        cols_to_drop = []
        if drop_group_name and drop_group_name in FEATURE_GROUPS:
            cols_to_drop = FEATURE_GROUPS[drop_group_name]
            cols_to_drop = [c for c in cols_to_drop if c in available_cols]
            if target_col in cols_to_drop:
                 cols_to_drop.remove(target_col)
        input_df = df.drop(columns=cols_to_drop, errors='ignore')
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaled_x = scaler_x.fit_transform(input_df)
    scaled_y = scaler_y.fit_transform(raw_y)
    
    X, y = [], []
    seq_len = CONFIG["seq_length"]
    
    for i in range(len(scaled_x) - seq_len):
        X.append(scaled_x[i : i + seq_len])
        y.append(scaled_y[i + seq_len, 0]) 
        
    X = np.array(X)
    y = np.array(y)
    
    dates = df.index[seq_len:]
    
    # Split
    test_start = pd.Timestamp(CONFIG["test_start_date"])
    test_end = pd.Timestamp(CONFIG["test_end_date"])
    
    train_mask = dates < test_start
    test_mask = (dates >= test_start) & (dates <= test_end)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_test = X[test_mask]
    y_test_raw = raw_y[seq_len:][test_mask] 
    test_dates = dates[test_mask]
    
    return X_train, y_train, X_test, y_test_raw, test_dates, scaler_y, len(input_df.columns)

# --- 3. Models ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, num_filters, kernel_size, seq_length):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        pooled_len = seq_length // 2
        self.fc = nn.Linear(num_filters * pooled_len, output_size)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        return self.fc(x.flatten(1))

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_filters, kernel_size):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(num_filters, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(out), dim=1) 
        context = torch.sum(attn_weights * out, dim=1) 
        out = self.fc(context)
        return out

# --- 4. Training ---
def train_model(model, X_train, y_train, config):
    X_t = torch.FloatTensor(X_train).to(config['device'])
    y_t = torch.FloatTensor(y_train).unsqueeze(1).to(config['device'])
    
    dataset = TensorDataset(X_t, y_t)
    # [수정] DataLoader의 shuffle을 False로 고정하면 순서가 보장되지만, 
    # 학습 효과를 위해 True를 유지하되 시드 고정으로 동일한 셔플링을 보장함.
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(config["epochs"]):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
    return model

# --- 5. Plotting Function ---
def plot_comparison(model_name, feature_config, dates, y_true, y_pred, rmse):
    plt.figure(figsize=(10, 6))
    
    # Actual
    plt.plot(dates, y_true, label='Actual', color='blue', marker='o', linewidth=2)
    
    # Predicted
    plt.plot(dates, y_pred, label='Predicted', color='red', linestyle='--', marker='x', linewidth=2)
    
    # Title with Optimization Info & RMSE
    if isinstance(feature_config, list):
        opt_str = "Optimized: Selected Features (LSTM+)"
    else:
        opt_str = f"Optimized: Drop {feature_config if feature_config else 'None'}"
        
    title_str = f"{model_name} Prediction ({opt_str})\nRMSE: {rmse:.4f}"
    plt.title(title_str, fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("KOSPI Index", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Date Formatting (No time)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()
    
    # Save
    safe_name = model_name.replace('+', '_').replace('(', '').replace(')', '')
    filename = f"Evaluation_{safe_name}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"   [Graph Saved] {filename}")
    plt.close()

# --- 6. Main Logic ---
def main():
    # [추가] 시드 고정 실행
    set_seed(CONFIG["seed"])
    
    print("Loading Data...")
    full_df = load_data(CONFIG)
    
    # 5 Scenarios
    scenarios = [
        ("CNN", "NASDAQ", CNNModel),
        ("LSTM", "VKOSPI", LSTMModel),
        ("CNN+LSTM", "KOSPI", CNNLSTMModel),
        ("LSTM(Attn)", "VKOSPI", LSTMAttentionModel),
        ("LSTM+", LSTM_PLUS_FEATURES, LSTMModel)
    ]
    
    print(f"\n[Start Evaluation] Period: {CONFIG['test_start_date']} ~ {CONFIG['test_end_date']}")
    
    rmse_results = []

    for model_name, feature_config, ModelClass in scenarios:
        # 모델별로 시드를 다시 초기화할지, 아니면 전체 시나리오를 하나의 시퀀스로 볼지 결정해야 함.
        # 모델 초기화 가중치가 동일하게 시작되도록 루프 내에서 시드 재설정하는 것이 좋음.
        set_seed(CONFIG["seed"]) 
        
        print(f"\n>> Processing {model_name}...")
        
        # 1. Prepare Data
        if isinstance(feature_config, list):
            X_train, y_train, X_test, y_test_raw, test_dates, scaler_y, input_dim = \
                process_features_and_split(full_df, select_features_list=feature_config)
        else:
            X_train, y_train, X_test, y_test_raw, test_dates, scaler_y, input_dim = \
                process_features_and_split(full_df, drop_group_name=feature_config)
        
        if len(X_test) == 0:
            print("   [Error] No test data found.")
            continue

        # 2. Init Model
        if model_name == "CNN":
            model = ModelClass(input_dim, 1, 32, 5, CONFIG["seq_length"]) 
        elif model_name == "CNN+LSTM":
            model = ModelClass(input_dim, 256, 1, 1, 32, 5)
        else:
            model = ModelClass(input_dim, 256, 1, 1)
            
        model.to(CONFIG['device'])
        
        # 3. Train
        model = train_model(model, X_train, y_train, CONFIG)
        
        # 4. Predict
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(X_test).to(CONFIG['device'])
            pred_scaled = model(input_tensor).cpu().numpy()
            
        # 5. Inverse Transform & RMSE
        pred_prices = scaler_y.inverse_transform(pred_scaled).flatten()
        y_true_prices = y_test_raw.flatten()
        
        rmse = np.sqrt(np.mean((y_true_prices - pred_prices) ** 2))
        rmse_results.append({"Model": model_name, "RMSE": rmse})
        print(f"   [Result] RMSE: {rmse:.4f}")
        
        # 6. Plot
        plot_comparison(model_name, feature_config, test_dates, y_true_prices, pred_prices, rmse)

    print("\n[RMSE Summary]")
    rmse_df = pd.DataFrame(rmse_results)
    print(rmse_df)
    rmse_df.to_csv("final_rmse_results.csv", index=False)

if __name__ == "__main__":
    main()