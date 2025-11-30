import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import random

# --- 설정 (Configuration) ---
CONFIG = {
    "data_file": "KOSPI_dataset_final.csv",
    # 11월 28일까지 학습 (미래 예측용)
    "data_start": "2013-08-06",
    "data_end": "2025-11-28",
    
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

# LSTM+ 모델용 피처 리스트 (Whitelist)
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

# --- [수동 Scaler] ---
class ManualScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None
        
    def fit_transform(self, data):
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0 # 0 나누기 방지
        return (data - self.min_) / self.range_
        
    def get_params(self):
        return {'min': self.min_, 'range': self.range_}

# --- 데이터 로드 및 전처리 ---
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
    df = df.ffill().bfill().dropna()
    df.columns = [c.strip() for c in df.columns]
    return df

def process_features(df, feature_config):
    target_col = "KOSPI_Close"
    available_cols = df.columns.tolist()
    
    # 1. Target (y)
    raw_y = df[[target_col]].values
    
    # 2. Input (X) 구성
    if isinstance(feature_config, list): # Whitelist (LSTM+)
        selected_cols = [c for c in feature_config if c in available_cols]
        input_df = df[selected_cols]
    else: # Blacklist (Drop Group)
        cols_to_drop = []
        if feature_config and feature_config in FEATURE_GROUPS:
            cols_to_drop = FEATURE_GROUPS[feature_config]
            cols_to_drop = [c for c in cols_to_drop if c in available_cols]
            if target_col in cols_to_drop: cols_to_drop.remove(target_col)
        input_df = df.drop(columns=cols_to_drop, errors='ignore')
    
    scaler_x = ManualScaler()
    scaler_y = ManualScaler()
    
    scaled_x = scaler_x.fit_transform(input_df.values)
    scaled_y = scaler_y.fit_transform(raw_y)
    
    X, y = [], []
    seq_len = CONFIG["seq_length"]
    horizon = CONFIG["predict_horizon"]
    
    valid_len = len(scaled_x) - seq_len - horizon + 1
    
    for i in range(valid_len):
        X.append(scaled_x[i : i + seq_len])
        y.append(scaled_y[i + seq_len : i + seq_len + horizon].flatten())
        
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler_x.get_params(), scaler_y.get_params(), len(input_df.columns), input_df.columns.tolist()

# --- 모델 정의 ---
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

def train_model(model, X_train, y_train, config):
    X_t = torch.FloatTensor(X_train).to(config['device'])
    y_t = torch.FloatTensor(y_train).to(config['device'])
    
    dataset = TensorDataset(X_t, y_t)
    # 시드 고정 효과를 위해 shuffle=True여도 순서가 동일하게 유지됨
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

def main():
    # [추가] 시드 고정 실행 (전체 프로세스 초기화)
    set_seed(CONFIG["seed"])
    
    print("Loading Data...")
    full_df = load_data(CONFIG)
    
    # 5가지 시나리오
    scenarios = [
        ("CNN", "NASDAQ", CNNModel),             # CNN: 나스닥 제거
        ("LSTM", "VKOSPI", LSTMModel),           # LSTM: VKOSPI 제거
        ("CNN+LSTM", "KOSPI", CNNLSTMModel),     # CNN+LSTM: KOSPI 제거
        ("LSTM_Attn", "VKOSPI", LSTMAttentionModel), # Attention: VKOSPI 제거
        ("LSTM+", LSTM_PLUS_FEATURES, LSTMModel) # LSTM+: 특정 피처 선택
    ]
    
    print(f"\n[Start Training & Saving Parameters] Seed: {CONFIG['seed']}")
    
    for model_name, feature_config, ModelClass in scenarios:
        # 모델별 초기화 상태도 동일하게 맞추기 위해 시드 재설정
        set_seed(CONFIG["seed"])
        
        print(f"\n>> Training {model_name}...")
        
        X, y, scaler_x_params, scaler_y_params, input_dim, feature_names = process_features(full_df, feature_config)
        
        # Model Init (Output=5)
        if model_name == "CNN":
            model = ModelClass(input_dim, CONFIG["predict_horizon"], 32, 5, CONFIG["seq_length"]) 
        elif model_name == "CNN+LSTM":
            model = ModelClass(input_dim, 256, 1, CONFIG["predict_horizon"], 32, 5)
        else:
            model = ModelClass(input_dim, 256, 1, CONFIG["predict_horizon"])
            
        model.to(CONFIG['device'])
        trained_model = train_model(model, X, y, CONFIG)
        
        save_path = f"{model_name}_params.pth"
        
        save_info = {
            "model_state_dict": trained_model.state_dict(),
            "scaler_x": scaler_x_params, 
            "scaler_y": scaler_y_params,
            "feature_names": feature_names,
            "input_dim": input_dim,
            "feature_config": feature_config
        }
        
        torch.save(save_info, save_path)
        print(f"   [Saved] {save_path}")

if __name__ == "__main__":
    main()
