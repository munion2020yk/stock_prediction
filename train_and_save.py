import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import os

# --- 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_PATH = "KOSPI_base.csv"
TIME_STEP = 60
HORIZON = 5
EPOCHS = 100 # 충분한 학습을 위해 설정

print(f"--- Device: {DEVICE} ---")

# --- 데이터 전처리 클래스 ---
class CustomMinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.scale_ = None
    
    def fit_transform(self, data):
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)
        self.scale_ = self.max_ - self.min_
        self.scale_[self.scale_ == 0] = 1.0
        return (data - self.min_) / self.scale_

# --- 모델 정의 ---
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNN_LSTM, self).__init__()
        # [Image of CNN LSTM architecture diagram]
        self.conv = nn.Conv1d(input_dim, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1) # (N, C, L)
        x = self.pool(self.relu(self.conv(x)))
        x = x.permute(0, 2, 1) # (N, L, C)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionLSTM, self).__init__()
        # [Image of Attention Mechanism in LSTM]
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        score = self.attention(outputs)
        weights = F.softmax(score, dim=1)
        context = torch.sum(outputs * weights, dim=1)
        return self.fc(context)

# --- 유틸리티 함수 ---
def load_data(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'cp949']
    df = pd.DataFrame()
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, sep='\t', index_col='Date', parse_dates=['Date'], encoding=enc)
            if len(df.columns) <= 1:
                df = pd.read_csv(file_path, sep=',', index_col='Date', parse_dates=['Date'], encoding=enc)
            break
        except: continue
    
    if not df.empty:
        df = df[df.index.notna()].sort_index().ffill().dropna()
        
        # Feature 선택 (기존 로직 유지)
        candidate_cols = ['KOSPI_Close', 'KOSPI_Open', 'KOSPI_High', 'KOSPI_Low', 'KOSPI_Volume',
                          'NAS_Close', 'NAS_Open', 'NAS_Volume', 'Rate', 'VKOSPI_close', 
                          'WTI_Close', 'USD_KRW_Close', 'KOSPI_future', 'Foreign_rate']
        cols = [c for c in candidate_cols if c in df.columns]
        return df[cols]
    return df

def create_sequences(data, time_step, horizon):
    X, y = [], []
    for i in range(len(data) - time_step - horizon + 1):
        X.append(data[i:(i + time_step), :])
        last_price = data[i + time_step - 1, 0] 
        future_prices = data[i + time_step : i + time_step + horizon, 0]
        y.append(future_prices - last_price) # 변동폭 학습
    return np.array(X), np.array(y)

def train_one_model(model, loader, epochs, name):
    print(f"[{name}] 학습 시작...")
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(epochs):
        avg_loss = 0
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss/len(loader):.6f}")
    return model

# --- 메인 실행 ---
def main():
    if not os.path.exists(FILE_PATH):
        print(f"오류: {FILE_PATH} 파일이 없습니다.")
        return

    # 1. 데이터 로드
    df = load_data(FILE_PATH)
    print(f"데이터 로드 완료: {df.shape}")
    
    # 2. 전처리
    scaler = CustomMinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    X, y = create_sequences(scaled_data, TIME_STEP, HORIZON)
    
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = df.shape[1]
    
    # 3. 모델 학습 및 저장
    models_to_train = [
        ("CNN_LSTM", CNN_LSTM(input_dim, 64, HORIZON), "cnn_lstm_model.pth"),
        ("Attention_LSTM", AttentionLSTM(input_dim, 64, HORIZON), "attn_lstm_model.pth")
    ]
    
    for name, model, filename in models_to_train:
        trained_model = train_one_model(model, loader, EPOCHS, name)
        
        # 저장할 정보: 모델 가중치 + 스케일러 정보 + 입력 차원
        save_info = {
            'model_state_dict': trained_model.state_dict(),
            'scaler_min': scaler.min_,
            'scaler_scale': scaler.scale_,
            'input_dim': input_dim,
            'feature_names': df.columns.tolist()
        }
        torch.save(save_info, filename)
        print(f"-> {name} 저장 완료: {filename}")

if __name__ == "__main__":
    main()
