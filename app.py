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

# 파일 경로 (파일명이 학습 코드와 일치해야 함)
DATA_FILE = "KOSPI_dataset_final.csv"
MODEL_FILES = {
    "LSTM": "LSTM_params.pth",
    "CNN": "CNN_params.pth",
    "CNN+LSTM": "CNN+LSTM_params.pth",
    "LSTM(Attention)": "LSTM(Attention)_params.pth"
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
    
    checkpoint = torch.load(pth_file, map_location=DEVICE, weights_only=False)
    input_dim = checkpoint['input_dim']
    horizon = 5
    
    if model_name == "CNN":
        model = CNNModel(input_dim, horizon, 32, 5, 5) # seq_len=5 fixed
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
    st.title("📈 KOSPI Prediction Service")
    st.markdown("딥러닝 모델을 활용한 **KOSPI 향후 5일 지수 예측** 서비스입니다.")

    # 폰트 크기 상태 관리
    if 'font_size' not in st.session_state:
        st.session_state.font_size = 20  # 기본 폰트 크기

    # 폰트 조절 버튼 (사이드바 또는 메인 상단)
    col_btn1, col_btn2, _ = st.columns([1, 1, 8])
    with col_btn1:
        if st.button("➕ 글자 크게"):
            st.session_state.font_size += 2
    with col_btn2:
        if st.button("➖ 글자 작게"):
            st.session_state.font_size = max(10, st.session_state.font_size - 2)

    # CSS 스타일 동적 적용
    st.markdown(f"""
        <style>
        div[data-testid="stDataFrame"] div[data-testid="stTable"] {{
            font-size: {st.session_state.font_size}px !important;
        }}
        div[data-testid="stDataFrame"] th {{
            font-size: {st.session_state.font_size}px !important;
        }}
        div[data-testid="stDataFrame"] td {{
            font-size: {st.session_state.font_size}px !important;
            line-height: 1.5 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    if not os.path.exists(DATA_FILE):
        st.error(f"데이터 파일({DATA_FILE})을 찾을 수 없습니다.")
        st.stop()
        
    df = load_csv_data(DATA_FILE)

    # --- 사이드바 설정 ---
    st.sidebar.header("설정 (Configuration)")
    
    model_options = ["LSTM", "CNN", "CNN+LSTM", "LSTM(Attention)"]
    selected_model_name = st.sidebar.radio("예측 모델 선택", model_options, index=0)
    
    loaded_data = load_model_checkpoint(selected_model_name)
    if loaded_data is None:
        st.sidebar.error(f"모델 파일({MODEL_FILES.get(selected_model_name)})이 없습니다. 학습을 먼저 진행하세요.")
        st.stop()
        
    model, checkpoint = loaded_data
    feature_names = checkpoint['feature_names']
    
    default_date = pd.Timestamp("2025-12-01").date()
    min_date = df.index.min().date() + pd.Timedelta(days=10)
    predict_date = st.sidebar.date_input("예측 시작 날짜", value=default_date, min_value=min_date)
    
    # --- 예측 실행 ---
    cutoff_date = pd.to_datetime(predict_date) - pd.Timedelta(days=1)
    
    try:
        input_df = df.loc[:cutoff_date, feature_names].tail(5)
    except KeyError:
        st.error("데이터 컬럼 불일치! 학습 데이터와 현재 데이터의 컬럼이 다릅니다.")
        st.stop()
        
    if len(input_df) < 5:
        st.error("과거 데이터 부족.")
        st.stop()
        
    # Scaling (X)
    scaler_x = checkpoint['scaler_x'] 
    input_raw = input_df.values
    input_scaled = (input_raw - scaler_x['min']) / scaler_x['range']
    
    # Predict
    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy().flatten()
        
    # Inverse Scaling (y)
    scaler_y = checkpoint['scaler_y']
    pred_prices = (pred_scaled * scaler_y['range']) + scaler_y['min']

    # --- 화면 구성 ---
    target_dates = pd.date_range(start=predict_date, periods=5, freq='B')
    date_strs = target_dates.strftime('%Y-%m-%d')
    
    st.subheader(f"📊 {selected_model_name} 예측 결과 ({predict_date} ~)")
    
    res_df = pd.DataFrame({
        "날짜": date_strs,
        "예측 지수 (Pt)": [f"{p:,.2f}" for p in pred_prices], 
        "등락": ["-" for _ in range(5)]
    })
    
    last_real_price = input_df["KOSPI_Close"].iloc[-1]
    diffs = []
    prev = last_real_price
    for p in pred_prices:
        d = p - prev
        sign = "🔺" if d > 0 else "🔻" if d < 0 else "-"
        diffs.append(f"{sign} {abs(d):.2f}")
        prev = p
    res_df["등락"] = diffs
    
    st.dataframe(res_df, use_container_width=True, hide_index=True, height=300)
    
    st.markdown("---")
    st.caption("📉 예측 추세 그래프 (참조용)")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        
        # [수정] 12월 1일 ~ 5일만 그리기 (이전 데이터 연결 X)
        plot_dates = target_dates
        plot_values = pred_prices
        
        ax.plot(plot_dates, plot_values, marker='o', color='red', linestyle='--', linewidth=1.5, label='Prediction')
        
        # 값 어노테이션 추가
        for i, (date, val) in enumerate(zip(plot_dates, plot_values)):
            ax.text(date, val, f"{val:.0f}", ha='center', va='bottom', color='red', fontsize=8, fontweight='bold')

        ax.set_title("5-Day Forecast Trend", fontsize=10)
        ax.set_ylabel("KOSPI Index")
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()
