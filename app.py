import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import os

# --- 설정 ---
st.set_page_config(page_title="KOSPI Prediction Service", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FILE = "KOSPI_dataset_final.csv"
MODEL_FILES = {
    "CNN+LSTM": "CNN+LSTM_params.pth",
    "LSTM+": "LSTM+_params.pth",
    "LSTM": "LSTM_params.pth",
    "CNN": "CNN_params.pth",
    "LSTM(Attention)": "LSTM_Attn_params.pth"
}

# --- 모델 클래스 ---
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

# --- 유틸리티 ---
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

def predict_with_model(model_name, full_df, cutoff_date):
    pth_file = MODEL_FILES.get(model_name)
    if not os.path.exists(pth_file): return None
    
    checkpoint = torch.load(pth_file, map_location=DEVICE, weights_only=False)
    input_dim = checkpoint['input_dim']
    feature_names = checkpoint['feature_names']
    horizon = 5
    
    # Init Model
    if model_name == "CNN":
        model = CNNModel(input_dim, horizon, 32, 5, 5) 
    elif model_name == "CNN+LSTM":
        model = CNNLSTMModel(input_dim, 256, 1, horizon, 32, 5)
    elif model_name == "LSTM(Attention)":
        model = LSTMAttentionModel(input_dim, 256, 1, horizon)
    else: # LSTM, LSTM+
        model = LSTMModel(input_dim, 256, 1, horizon)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Prepare Data
    try:
        input_df = full_df.loc[:cutoff_date, feature_names].tail(5)
    except KeyError:
        return None
        
    if len(input_df) < 5: return None
    
    # Scaling X
    scaler_x = checkpoint['scaler_x']
    input_raw = input_df.values
    input_scaled = (input_raw - scaler_x['min']) / scaler_x['range']
    
    # Predict
    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy().flatten()
        
    # Inverse Y
    scaler_y = checkpoint['scaler_y']
    pred_prices = (pred_scaled * scaler_y['range']) + scaler_y['min']
    
    return pred_prices

# --- 메인 앱 ---
def main():
    st.title("📈 KOSPI Prediction Service")
    st.markdown("딥러닝 모델을 활용한 **KOSPI 향후 5일 지수 예측** 서비스입니다.")

    if not os.path.exists(DATA_FILE):
        st.error(f"데이터 파일({DATA_FILE}) 없음.")
        st.stop()
    
    df = load_csv_data(DATA_FILE)
    
    # --- 사이드바 ---
    st.sidebar.header("설정 (Settings)")
    
    # 날짜 선택
    default_date = pd.Timestamp("2025-12-01").date()
    min_date = df.index.min().date() + pd.Timedelta(days=10)
    predict_date = st.sidebar.date_input("예측 시작 날짜", value=default_date, min_value=min_date)
    cutoff_date = pd.to_datetime(predict_date) - pd.Timedelta(days=1)
    
    st.sidebar.markdown("---")
    
    # [수정] 라디오 박스 모델 선택 (CNN+LSTM 최상단)
    # 순서: CNN+LSTM -> LSTM+ -> LSTM -> CNN -> LSTM(Attention)
    model_options = ["CNN+LSTM", "LSTM+", "LSTM", "CNN", "LSTM(Attention)"]
    selected_model_name = st.sidebar.radio("예측 모델 선택", model_options, index=0)
    
    # --- 메인 로직 ---
    
    # 선택된 모델 예측 수행
    pred_prices = predict_with_model(selected_model_name, df, cutoff_date)
    
    if pred_prices is None:
        st.error(f"'{selected_model_name}' 예측에 실패했습니다. 데이터 기간이나 모델 파일을 확인해주세요.")
        st.stop()
        
    target_dates = pd.date_range(start=predict_date, periods=5, freq='B')
    date_strs = target_dates.strftime('%Y-%m-%d')
    
    # 전일 종가 (등락 계산용)
    last_real_price = df["KOSPI_Close"].loc[:cutoff_date].iloc[-1]
    
    # 1. 결과 표시 (텍스트)
    st.subheader(f"🚀 {selected_model_name} 예측 결과 ({predict_date} ~)")
    
    cols = st.columns(5)
    prev_price = last_real_price
    
    for i, (col, date, price) in enumerate(zip(cols, date_strs, pred_prices)):
        diff = price - prev_price
        
        if diff > 0:
            color = "#d62728" # 빨강
            arrow = "▲"
        elif diff < 0:
            color = "#1f77b4" # 파랑
            arrow = "▼"
        else:
            color = "gray"
            arrow = "-"
            
        with col:
            st.markdown(f"""
                <div style="text-align: center; border: 1px solid #eee; border-radius: 10px; padding: 10px;">
                    <div style="font-size: 14px; color: gray;">{date}</div>
                    <div style="font-size: 24px; font-weight: bold;">{price:,.2f}</div>
                    <div style="font-size: 16px; color: {color}; font-weight: bold;">
                        {arrow} {abs(diff):.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        prev_price = price
        
    # 2. 그래프
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("📉 예측 추세 그래프")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        plot_dates = target_dates
        plot_values = pred_prices
        
        # 선택된 모델 그래프
        ax.plot(plot_dates, plot_values, marker='o', color='#d62728', linestyle='-', linewidth=3, label=selected_model_name)
        
        # 값 표시
        for date, val in zip(plot_dates, plot_values):
            ax.text(date, val, f"{val:.0f}", ha='center', va='bottom', color='#d62728', fontsize=9, fontweight='bold')
            
        # 기준선 (Ref Price)
        ax.axhline(y=last_real_price, color='gray', linestyle=':', linewidth=1, label=f'Ref: {last_real_price:,.0f}')
        
        ax.set_ylabel("KOSPI Index")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
    with col2:
        st.info(f"ℹ️ {selected_model_name} 모델 정보")
        
        descriptions = {
            "CNN+LSTM": "CNN으로 특징을 추출하고 LSTM으로 시계열을 학습한 하이브리드 모델입니다.",
            "LSTM+": "핵심 피처(KOSPI OHLCV, 선물, 환율 등)만 선별하여 학습한 고성능 모델입니다.",
            "LSTM": "전통적인 시계열 예측 모델로, 전체 피처(VKOSPI 제외)를 사용합니다.",
            "CNN": "1D Convolution을 사용하여 시계열 데이터의 국소적 패턴을 학습합니다.",
            "LSTM(Attention)": "Attention 메커니즘을 적용하여 중요한 시점에 가중치를 둡니다."
        }
        st.write(descriptions.get(selected_model_name, ""))

if __name__ == "__main__":
    main()
