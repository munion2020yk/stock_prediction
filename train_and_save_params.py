import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

# --- 설정 (Configuration) ---
CONFIG = {
    "data_file": "KOSPI_dataset_final.csv",
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
    "learning_rate": 0.001,
    
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- Feature Groups ---
FEATURE_GROUPS = {
    "KOSPI": ['KOSPI_Close', 'KOSPI_Open', 'KOSPI_High', 'KOSPI_Low', 'KOSPI_Volume', 'KOSPI_Amount', 'KOSPI_Change', 'KOSPI_Fluctuation', 'KOSPI_UpDown'],
    "Foreign": ['Foreign_MarketCap_Ratio', 'Foreign_MarketCap', 'Foreign_Rate'],
}

print(f"Using Device: {CONFIG['device']}")

# --- [수정] 명확한 수동 Scaler 클래스 ---
class ManualScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None
        
    def fit_transform(self, data):
        # data: numpy array (N, Features)
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)
        self.range_ = self.max_ - self.min_
        
        # 0으로 나누기 방지
        self.range_[self.range_ == 0] = 1.0
        
        return (data - self.min_) / self.range_
        
    def get_params(self):
        return {
            'min': self.min_,
            'range': self.range_
        }

# --- 데이터 로드 함수 ---
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

def process_features(df, drop_group_name=None):
    target_col = "KOSPI_Close"
    available_cols = df.columns.tolist()
    
    cols_to_drop = []
    if drop_group_name and drop_group_name in FEATURE_GROUPS:
        cols_to_drop = FEATURE_GROUPS[drop_group_name]
        cols_to_drop = [c for c in cols_to_drop if c in available_cols]
        if target_col in cols_to_drop:
             cols_to_drop.remove(target_col)
    
    # Target (y)
    raw_y = df[[target_col]].values
    
    # Input (X)
    input_df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # [수정] ManualScaler 사용
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
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(config["epochs"]):
        epoch_loss = 0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}, Loss: {epoch_loss/len(loader):.6f}")
            
    return model

def main():
    print("Loading Data...")
    full_df = load_data(CONFIG)
    
    scenarios = [
        ("CNN", "Foreign", CNNModel),
        ("LSTM", None, LSTMModel),
        ("CNN+LSTM", "KOSPI", CNNLSTMModel),
        ("LSTM(Attention)", "Foreign", LSTMAttentionModel) # 이름 통일
    ]
    
    print("\n[Start Training & Saving Parameters]")
    
    for model_name, drop_group, ModelClass in scenarios:
        print(f"\n>> Training {model_name} (Drop: {drop_group})...")
        
        X, y, scaler_x_params, scaler_y_params, input_dim, feature_names = process_features(full_df, drop_group)
        
        if model_name == "CNN":
            model = ModelClass(input_dim, 5, 32, 5, CONFIG["seq_length"]) 
        elif model_name == "CNN+LSTM":
            model = ModelClass(input_dim, 256, 1, 5, 32, 5)
        else:
            model = ModelClass(input_dim, 256, 1, 5)
            
        model.to(CONFIG['device'])
        trained_model = train_model(model, X, y, CONFIG)
        
        save_path = f"{model_name}_params.pth"
        
        # [수정] 수동 Scaler 파라미터 저장
        save_info = {
            "model_state_dict": trained_model.state_dict(),
            "scaler_x": scaler_x_params, # {'min': array, 'range': array}
            "scaler_y": scaler_y_params, # {'min': array, 'range': array}
            "feature_names": feature_names,
            "input_dim": input_dim,
            "drop_group": drop_group
        }
        
        torch.save(save_info, save_path)
        print(f"   [Saved] {save_path}")

if __name__ == "__main__":
    main()