import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- 설정 ---
st.set_page_config(page_title="KOSPI Prediction App", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파일 경로 (같은 폴더 기준)
DATA_FILE = "KOSPI_dataset_final.csv"
MODEL_FILES = {
    "LSTM": "LSTM_params.pth",
    "CNN": "CNN_params.pth",
    "CNN+LSTM": "CNN+LSTM_params.pth",
    "LSTM(Attention)": "LSTM_Attn_params.pth"
}

# --- 모델 클래스 정의 (학습 코드와 동일) ---
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

# --- 유틸리티 함수 ---
@st.cache_data
def load_csv_data(filepath):
    if not os.path.exists(filepath): return pd.DataFrame()
    encodings = ['utf-16', 'utf-8', 'utf-8-sig', 'cp949', 'latin1']
    df = None
    for enc in encodings:
        try:
            temp_df = pd.read_csv(filepath, sep='\t', index_col="Date", parse_dates=True, encoding=enc)
            if len(temp_df.columns) > 1: df = temp_df; break
            temp_df = pd.read_csv(filepath, sep=',', index_col="Date", parse_dates=True, encoding=enc)
            if len(temp_df.columns) > 1: df = temp_df; break
        except: continue
    
    if df is not None:
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.ffill().bfill().dropna()
        df.columns = [c.strip() for c in df.columns]
    return df

def load_model_checkpoint(model_name):
    pth_file = MODEL_FILES.get(model_name)
    if not os.path.exists(pth_file): return None
    
    # weights_only=False 필수 (numpy, dict 포함)
    checkpoint = torch.load(pth_file, map_location=DEVICE, weights_only=False)
    
    input_dim = checkpoint['input_dim']
    seq_len = 5 # Configured in training
    horizon = 5 # Configured in training
    
    # Init Model
    if model_name == "CNN":
        model = CNNModel(input_dim, horizon, 32, 5, seq_len)
    elif model_name == "CNN+LSTM":
        model = CNNLSTMModel(input_dim, 256, 1, horizon, 32, 5)
    elif model_name == "LSTM(Attention)":
        model = LSTMAttentionModel(input_dim, 256, 1, horizon)
    else: # LSTM
        model = LSTMModel(input_dim, 256, 1, horizon)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    return model, checkpoint

# --- 메인 앱 ---
def main():
    st.title("📈 KOSPI Stock Prediction Service")
    st.markdown("딥러닝 모델을 활용한 **KOSPI 향후 5일 주가 예측** 서비스입니다.")

    # 1. 데이터 로드 확인
    df = load_csv_data(DATA_FILE)
    if df.empty:
        st.error(f"데이터 파일({DATA_FILE})을 찾을 수 없습니다.")
        st.stop()

    # --- 사이드바 설정 ---
    st.sidebar.header("설정 (Configuration)")
    
    # 2. 모델 선택 (Radio Box)
    model_options = ["LSTM", "CNN", "CNN+LSTM", "LSTM(Attention)"]
    selected_model_name = st.sidebar.radio("예측 모델 선택", model_options, index=0) # 초기값 LSTM
    
    # 모델 로드
    loaded_data = load_model_checkpoint(selected_model_name)
    if loaded_data is None:
        st.sidebar.error(f"모델 파일({MODEL_FILES.get(selected_model_name)})이 없습니다.")
        st.stop()
        
    model, checkpoint = loaded_data
    feature_names = checkpoint['feature_names']
    
    # 3. 날짜 선택
    last_date = df.index[-1]
    default_date = pd.Timestamp("2025-12-01").date()
    min_date = df.index.min().date() + pd.Timedelta(days=10)
    
    predict_date = st.sidebar.date_input("예측 시작 날짜", value=default_date, min_value=min_date)
    
    # --- 예측 실행 및 결과 표시 ---
    
    # 데이터 준비 (cutoff date 기준 과거 5일)
    cutoff_date = pd.to_datetime(predict_date) - pd.Timedelta(days=1)
    
    # 학습 때 사용한 Feature만 선택
    try:
        input_df = df.loc[:cutoff_date, feature_names].tail(5)
    except KeyError:
        st.error("데이터 컬럼이 모델 학습 시와 다릅니다.")
        st.stop()
        
    if len(input_df) < 5:
        st.error("과거 데이터가 부족하여 예측할 수 없습니다.")
        st.stop()
        
    # Scaling (X) - 저장된 scaler 파라미터 사용
    x_min = checkpoint['scaler_x_min']
    x_scale = checkpoint['scaler_x_scale']
    
    input_raw = input_df.values
    input_scaled = (input_raw - x_min) / x_scale
    
    # Predict
    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy().flatten()
        
    # Inverse Scaling (y)
    y_params = checkpoint['scaler_y_params']
    y_min = y_params['min'][0]
    y_scale = y_params['scale'][0]
    
    pred_prices = (pred_scaled * y_scale) + y_params['data_min'][0] # min_ + data_min 주의 (sklearn 구조에 따름)
    # sklearn minmax: X_std = (X - X.min) / (X.max - X.min)
    # X_scaled = X_std * (max - min) + min
    # Inverse: X = X_scaled * scale_ + min_ 
    # checkpoint 저장시 scaler.min_ 과 scale_ 저장했음.
    # 정확한 역변환: (val - min_) / scale_  (X) -> val * scale + min? (X)
    # sklearn 공식: X = (X_scaled - min_) / scale_  (X)
    # -> X_scaled = X * scale_ + min_
    # -> X = (X_scaled - min_) / scale_
    pred_prices = (pred_scaled - y_params['min'][0]) / y_params['scale'][0]

    # --- 화면 구성 ---
    
    # 날짜 생성
    target_dates = pd.date_range(start=predict_date, periods=5, freq='B')
    date_strs = target_dates.strftime('%Y-%m-%d')
    
    # 1. 숫자 테이블 (크게)
    st.subheader(f"📊 {selected_model_name} 예측 결과 ({predict_date} ~)")
    
    res_df = pd.DataFrame({
        "날짜": date_strs,
        "예측 주가 (KRW)": [f"{p:,.0f}" for p in pred_prices],
        "등락": ["-" for _ in range(5)] # 전일대비 계산 가능하면 추가
    })
    
    # 전일 대비 계산
    last_real_price = input_df["KOSPI_Close"].iloc[-1]
    diffs = []
    prev = last_real_price
    for p in pred_prices:
        d = p - prev
        sign = "🔺" if d > 0 else "🔻" if d < 0 else "-"
        diffs.append(f"{sign} {abs(d):.0f}")
        prev = p
    res_df["등락"] = diffs
    
    # 테이블 스타일링 (글자 크기 키우기)
    st.dataframe(res_df, use_container_width=True, hide_index=True)
    
    # 2. 참조용 그래프 (작게)
    st.markdown("---")
    st.caption("📉 예측 추세 그래프 (참조용)")
    
    col1, col2, col3 = st.columns([1, 2, 1]) # 가운데 정렬 효과
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3)) # 작은 사이즈
        
        # 시작점 (과거 1일) 연결
        plot_dates = [input_df.index[-1]] + list(target_dates)
        plot_values = [last_real_price] + list(pred_prices)
        
        ax.plot(plot_dates, plot_values, marker='o', color='red', linestyle='--', linewidth=1.5, label='Prediction')
        ax.axhline(y=last_real_price, color='gray', linestyle=':', linewidth=1, label='Ref Price')
        
        ax.set_title("5-Day Forecast Trend", fontsize=10)
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()
