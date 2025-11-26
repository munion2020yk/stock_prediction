import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
import os

# --- 설정 ---
st.set_page_config(page_title="Stock Prediction Inference", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_STEP = 60
HORIZON = 5

# --- 모델 클래스 정의 (학습 코드와 동일해야 함) ---
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNN_LSTM, self).__init__()
        self.conv = nn.Conv1d(input_dim, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv(x)))
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionLSTM, self).__init__()
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
class CustomMinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.scale_ = None
    
    def load_params(self, min_val, scale_val):
        self.min_ = min_val
        self.scale_ = scale_val

    def transform(self, data):
        return (data - self.min_) / self.scale_
    
    def inverse_transform_col(self, data, col_index):
        return (data * self.scale_[col_index]) + self.min_[col_index]

@st.cache_data
def load_csv_data(uploaded_file):
    encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'cp949']
    df = pd.DataFrame()
    bytes_data = uploaded_file.getvalue()
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(bytes_data), sep='\t', index_col='Date', parse_dates=['Date'], encoding=enc)
            if len(df.columns) <= 1:
                df = pd.read_csv(io.BytesIO(bytes_data), sep=',', index_col='Date', parse_dates=['Date'], encoding=enc)
            break
        except: continue
    
    if not df.empty:
        df = df[df.index.notna()].sort_index().ffill().dropna()
    return df

def load_checkpoint(uploaded_file, model_class):
    # 메모리 버퍼에서 로드
    checkpoint = torch.load(io.BytesIO(uploaded_file.getvalue()), map_location=DEVICE)
    
    input_dim = checkpoint['input_dim']
    scaler = CustomMinMaxScaler()
    scaler.load_params(checkpoint['scaler_min'], checkpoint['scaler_scale'])
    
    model = model_class(input_dim, 64, HORIZON)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    return model, scaler, input_dim, checkpoint.get('feature_names', [])

# --- 메인 앱 ---
def main():
    st.title("⚡ 빠른 주가 예측 (Inference Mode)")
    st.markdown("미리 학습된 `.pth` 파일을 업로드하여 대기 시간 없이 즉시 예측 결과를 확인하세요.")

    # 사이드바
    st.sidebar.header("1. 파일 업로드")
    
    # 1) 데이터 파일
    data_file = st.sidebar.file_uploader("KOSPI 데이터 (csv)", type=['csv', 'txt'])
    
    # 2) 모델 파라미터 파일
    st.sidebar.markdown("---")
    st.sidebar.subheader("학습된 파라미터 (.pth)")
    model_cnn_file = st.sidebar.file_uploader("메인: CNN+LSTM (.pth)", type=['pth'], key='cnn')
    model_attn_file = st.sidebar.file_uploader("보조: Attention LSTM (.pth)", type=['pth'], key='attn')

    # 메인 로직
    if data_file is not None:
        df = load_csv_data(data_file)
        st.sidebar.success(f"데이터 로드 완료 ({len(df)} rows)")
        
        # 모델 선택 체크박스
        st.sidebar.markdown("---")
        st.sidebar.subheader("2. 예측 모델 선택")
        use_cnn = st.sidebar.checkbox("메인: CNN+LSTM", value=True, disabled=(model_cnn_file is None))
        use_attn = st.sidebar.checkbox("보조: Attention LSTM", value=False, disabled=(model_attn_file is None))

        if not (use_cnn or use_attn):
            st.warning("최소 하나의 모델 파라미터 파일을 업로드하고 체크박스를 선택해주세요.")
            return

        # 날짜 선택
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. 예측 시점 설정")
        
        # Default 12월 1일 설정
        default_date = pd.Timestamp("2025-12-01").date()
        min_date = df.index.min().date() + pd.Timedelta(days=60)
        max_date = df.index.max().date() + pd.Timedelta(days=1)
        
        # 범위 보정
        if default_date > max_date: default_date = max_date
        if default_date < min_date: default_date = min_date

        predict_date = st.date_input("예측 시작 날짜 (이 날짜부터 5일)", value=default_date, min_value=min_date)

        # 예측 실행 버튼
        if st.button("🔮 예측 실행", type="primary"):
            
            # 입력 데이터 준비 (과거 60일)
            cutoff_date = pd.to_datetime(predict_date) - pd.Timedelta(days=1)
            
            # 파라미터 파일에서 feature names를 가져와서 컬럼 순서 맞추기 (중요)
            # CNN 모델이 있다면 CNN 기준, 없다면 Attn 기준
            ref_file = model_cnn_file if model_cnn_file else model_attn_file
            temp_ckpt = torch.load(io.BytesIO(ref_file.getvalue()), map_location=DEVICE)
            feature_cols = temp_ckpt.get('feature_names', df.columns.tolist())
            
            # 컬럼 필터링 (없는 컬럼 있으면 에러 처리 필요하지만 여기선 try)
            try:
                input_df = df.loc[:cutoff_date, feature_cols].tail(TIME_STEP)
            except KeyError:
                st.error(f"CSV 파일의 컬럼이 학습 데이터와 다릅니다. 필요 컬럼: {feature_cols}")
                return

            if len(input_df) < TIME_STEP:
                st.error("과거 데이터가 부족합니다.")
                return

            # 예측 로직
            results = {}
            
            # 1. CNN+LSTM 예측
            if use_cnn and model_cnn_file:
                model, scaler, _, _ = load_checkpoint(model_cnn_file, CNN_LSTM)
                
                # 전처리
                input_raw = input_df.values
                input_scaled = scaler.transform(input_raw)
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # 추론
                with torch.no_grad():
                    pred_change = model(input_tensor).cpu().numpy().flatten()
                
                # 복원
                last_val_scaled = input_scaled[-1, 0]
                pred_val_scaled = pred_change + last_val_scaled
                pred_final = scaler.inverse_transform_col(pred_val_scaled, 0)
                results["CNN+LSTM"] = pred_final

            # 2. Attention LSTM 예측
            if use_attn and model_attn_file:
                model, scaler, _, _ = load_checkpoint(model_attn_file, AttentionLSTM)
                
                input_raw = input_df.values
                input_scaled = scaler.transform(input_raw)
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    pred_change = model(input_tensor).cpu().numpy().flatten()
                
                last_val_scaled = input_scaled[-1, 0]
                pred_val_scaled = pred_change + last_val_scaled
                pred_final = scaler.inverse_transform_col(pred_val_scaled, 0)
                results["Attention LSTM"] = pred_final

            # --- 결과 시각화 ---
            st.divider()
            st.subheader(f"📅 예측 결과 ({predict_date} ~ 5일간)")
            
            # 날짜 생성
            target_dates = pd.date_range(start=predict_date, periods=HORIZON, freq='B')
            date_strs = target_dates.strftime('%Y-%m-%d')
            
            res_df = pd.DataFrame({"날짜": date_strs})
            for name, val in results.items():
                res_df[name] = np.round(val, 2)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("##### 예측값 테이블")
                st.dataframe(res_df, hide_index=True)
                
            with col2:
                st.write("##### 예측 그래프")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                colors = {"CNN+LSTM": "red", "Attention LSTM": "blue"}
                styles = {"CNN+LSTM": "-", "Attention LSTM": "--"}
                
                for name, val in results.items():
                    ax.plot(res_df['날짜'], val, label=name, 
                            color=colors.get(name, "gray"), 
                            linestyle=styles.get(name, "-"), marker='o')
                
                # 과거 데이터 (문맥용)
                past_days = 15
                past_data = df.loc[:cutoff_date, feature_cols[0]].tail(past_days)
                ax.plot(past_data.index.strftime('%Y-%m-%d'), past_data.values, color='gray', alpha=0.3, label='History')
                
                ax.set_title("Prediction Result")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)

    else:
        st.info("왼쪽 사이드바에서 데이터 및 모델 파일을 업로드해주세요.")

if __name__ == "__main__":
    main()
