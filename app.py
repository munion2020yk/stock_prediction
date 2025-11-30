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
    "LSTM+": "LSTM+_params.pth",
    "LSTM": "LSTM_params.pth",
    "CNN": "CNN_params.pth",
    "CNN+LSTM": "CNN+LSTM_params.pth",
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
    
    # weights_only=False 필수
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
    st.markdown("핵심 모델 **LSTM+**를 중심으로 다양한 딥러닝 모델의 예측 결과를 비교 분석합니다.")

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
    st.sidebar.header("모델 비교 (Comparison)")
    st.sidebar.info("메인 모델(LSTM+)은 항상 표시됩니다.")
    
    # 체크박스 (나머지 4개 모델)
    show_lstm = st.sidebar.checkbox("LSTM", value=False)
    show_cnn = st.sidebar.checkbox("CNN", value=False)
    show_cnnlstm = st.sidebar.checkbox("CNN+LSTM", value=False)
    show_attn = st.sidebar.checkbox("LSTM(Attention)", value=False)
    
    # --- 메인 로직 ---
    
    # 1. 메인 모델 (LSTM+) 예측
    pred_lstm_plus = predict_with_model("LSTM+", df, cutoff_date)
    
    if pred_lstm_plus is None:
        st.error("예측에 실패했습니다. 데이터 기간이나 모델 파일을 확인해주세요.")
        st.stop()
        
    target_dates = pd.date_range(start=predict_date, periods=5, freq='B')
    date_strs = target_dates.strftime('%Y-%m-%d')
    
    # 등락 비교를 위한 이전 종가 (11월 28일)
    last_real_price = df["KOSPI_Close"].loc[:cutoff_date].iloc[-1]
    
    # 2. 결과 표시 (텍스트) - LSTM+ 기준
    st.subheader(f"🚀 LSTM+ 예측 결과 ({predict_date} ~)")
    
    cols = st.columns(5)
    prev_price = last_real_price
    
    for i, (col, date, price) in enumerate(zip(cols, date_strs, pred_lstm_plus)):
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
        
    # 3. 그래프 (비교 기능 포함)
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("📉 모델별 예측 추세 비교 (시작일 기준)")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # [수정] 그래프 데이터: 12월 1일 ~ 12월 5일만 사용 (이전 데이터 연결 X)
        plot_dates = target_dates
        
        # 1) LSTM+ (메인, 굵은 빨강)
        val_plus = pred_lstm_plus # 5개 값
        ax.plot(plot_dates, val_plus, marker='o', color='#d62728', linestyle='-', linewidth=3, label='LSTM+ (Main)')
        
        # 값 표시 (메인 모델만)
        for date, val in zip(plot_dates, val_plus):
            ax.text(date, val, f"{val:.0f}", ha='center', va='bottom', color='#d62728', fontsize=9, fontweight='bold')
            
        # 2) 비교 모델들 (얇은 점선)
        compare_models = []
        if show_lstm: compare_models.append("LSTM")
        if show_cnn: compare_models.append("CNN")
        if show_cnnlstm: compare_models.append("CNN+LSTM")
        if show_attn: compare_models.append("LSTM(Attention)")
        
        colors = {'LSTM': 'blue', 'CNN': 'green', 'CNN+LSTM': 'orange', 'LSTM(Attention)': 'purple'}
        
        for name in compare_models:
            pred = predict_with_model(name, df, cutoff_date)
            if pred is not None:
                # 5개 예측값만 사용
                ax.plot(plot_dates, pred, marker='x', color=colors.get(name, 'gray'), linestyle='--', linewidth=1.5, label=name, alpha=0.7)

        # 기준선 (Ref Price: 11월 28일 종가)
        ax.axhline(y=last_real_price, color='gray', linestyle=':', linewidth=1, label=f'Ref: {last_real_price:,.0f}')
        
        ax.set_ylabel("KOSPI Index")
        # 날짜 포맷팅
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
    with col2:
        st.info("ℹ️ 모델 설명")
        st.markdown("""
        * **LSTM+**: 핵심 피처(KOSPI OHLCV, 선물, 환율 등)만 선별하여 학습한 고성능 모델
        * **LSTM**: 전체 피처 사용 (VKOSPI 제외)
        * **CNN**: 나스닥 제외, 합성곱 신경망
        * **CNN+LSTM**: KOSPI 지수 제외 하이브리드
        * **LSTM(Attn)**: 어텐션 메커니즘 적용
        """)

if __name__ == "__main__":
    main()
