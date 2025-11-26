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

# 기본 파일 경로 설정 (같은 폴더에 있다고 가정)
DATA_FILE_PATH = "KOSPI_base.csv"
CNN_MODEL_PATH = "cnn_lstm_model.pth"
ATTN_MODEL_PATH = "attn_lstm_model.pth"

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
def load_csv_data(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'cp949']
    df = pd.DataFrame()
    
    if not os.path.exists(file_path):
        return df

    for enc in encodings:
        try:
            df = pd.read_csv(file_path, sep='\t', index_col='Date', parse_dates=['Date'], encoding=enc)
            if len(df.columns) <= 1:
                df = pd.read_csv(file_path, sep=',', index_col='Date', parse_dates=['Date'], encoding=enc)
            break
        except: continue
    
    if not df.empty:
        df = df[df.index.notna()].sort_index().ffill().dropna()
    return df

def load_checkpoint(file_path, model_class):
    # 파일 경로에서 직접 로드
    try:
        checkpoint = torch.load(file_path, map_location=DEVICE)
    except Exception as e:
        st.error(f"모델 파일 로드 실패: {file_path}")
        st.warning("파일이 손상되었거나 Git LFS 포인터일 수 있습니다.")
        raise e
    
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
    st.title("⚡ 주가 예측 자동 분석 (Inference Mode)")
    st.markdown("서버에 저장된 데이터와 학습된 모델을 **자동으로 로드**하여 분석합니다.")

    # 사이드바 상태 표시
    st.sidebar.header("시스템 상태 (자동 로드)")

    # 1. 데이터 로드 확인
    if os.path.exists(DATA_FILE_PATH):
        df = load_csv_data(DATA_FILE_PATH)
        if not df.empty:
            st.sidebar.success(f"✅ 데이터 로드됨 ({len(df)}일)")
        else:
            st.sidebar.error("❌ 데이터 파일 읽기 실패")
            st.stop()
    else:
        st.sidebar.error(f"❌ 데이터 파일 없음: {DATA_FILE_PATH}")
        st.info(f"실행 경로에 '{DATA_FILE_PATH}' 파일이 있는지 확인해주세요.")
        st.stop()

    # 2. 모델 파일 확인
    cnn_exists = os.path.exists(CNN_MODEL_PATH)
    attn_exists = os.path.exists(ATTN_MODEL_PATH)
    
    if cnn_exists:
        st.sidebar.success(f"✅ CNN+LSTM 모델 발견")
    else:
        st.sidebar.warning(f"⚠️ CNN+LSTM 모델 없음 ({CNN_MODEL_PATH})")

    if attn_exists:
        st.sidebar.success(f"✅ Attention LSTM 모델 발견")
    else:
        st.sidebar.warning(f"⚠️ Attention LSTM 모델 없음 ({ATTN_MODEL_PATH})")

    if not (cnn_exists or attn_exists):
        st.error("사용 가능한 모델 파일(.pth)이 없습니다. 'train_and_save.py'를 먼저 실행해주세요.")
        st.stop()

    # 3. 모델 선택
    st.sidebar.markdown("---")
    st.sidebar.subheader("예측 모델 설정")
    use_cnn = st.sidebar.checkbox("메인: CNN+LSTM", value=cnn_exists, disabled=not cnn_exists)
    use_attn = st.sidebar.checkbox("보조: Attention LSTM", value=attn_exists, disabled=not attn_exists)

    if not (use_cnn or use_attn):
        st.warning("최소 하나의 모델을 선택해주세요.")
        return

    # 4. 날짜 선택
    st.sidebar.markdown("---")
    st.sidebar.subheader("예측 시점 설정")
    
    # Default 12월 1일 설정
    default_date = pd.Timestamp("2025-12-01").date()
    min_date = df.index.min().date() + pd.Timedelta(days=60)
    max_date = df.index.max().date() + pd.Timedelta(days=1)
    
    if default_date > max_date: default_date = max_date
    if default_date < min_date: default_date = min_date

    predict_date = st.date_input("예측 시작 날짜 (이 날짜부터 5일)", value=default_date, min_value=min_date)

    # 메인 화면 구성
    st.markdown("---")
    
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🔮 예측 실행", type="primary")
    with col_info:
        st.write(f"선택된 기준일: **{predict_date}** (데이터 마지막 날짜: {df.index.max().date()})")

    # 예측 실행 로직
    if run_btn:
        # 입력 데이터 준비
        cutoff_date = pd.to_datetime(predict_date) - pd.Timedelta(days=1)
        
        # 참조 모델 파일 결정 (컬럼 매핑용)
        ref_path = CNN_MODEL_PATH if cnn_exists else ATTN_MODEL_PATH
        
        try:
            temp_ckpt = torch.load(ref_path, map_location=DEVICE)
        except Exception as e:
            st.error(f"모델 파라미터 로드 실패: {ref_path}")
            st.error(f"Error Details: {e}")
            st.warning("Tip: .pth 파일이 정상적인 바이너리 파일인지 확인하세요. (Git LFS Pointer일 가능성 있음)")
            return

        feature_cols = temp_ckpt.get('feature_names', df.columns.tolist())
        
        # 데이터 슬라이싱
        try:
            # cutoff_date까지의 데이터 중 마지막 60개
            input_df = df.loc[:cutoff_date, feature_cols].tail(TIME_STEP)
        except KeyError:
            st.error(f"데이터 컬럼 불일치. 모델 학습 시 사용된 컬럼: {feature_cols}")
            return

        if len(input_df) < TIME_STEP:
            st.error(f"과거 데이터가 부족합니다. (필요: 60일, 실제: {len(input_df)}일)")
            return

        # 예측 수행
        results = {}
        
        # 1. CNN+LSTM
        if use_cnn and cnn_exists:
            try:
                model, scaler, _, _ = load_checkpoint(CNN_MODEL_PATH, CNN_LSTM)
                input_raw = input_df.values
                input_scaled = scaler.transform(input_raw)
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    pred_change = model(input_tensor).cpu().numpy().flatten()
                
                last_val_scaled = input_scaled[-1, 0] # 0번 컬럼 Target 가정
                pred_val_scaled = pred_change + last_val_scaled
                pred_final = scaler.inverse_transform_col(pred_val_scaled, 0)
                results["CNN+LSTM"] = pred_final
            except Exception as e:
                st.error(f"CNN+LSTM 예측 중 오류 발생: {e}")

        # 2. Attention LSTM
        if use_attn and attn_exists:
            try:
                model, scaler, _, _ = load_checkpoint(ATTN_MODEL_PATH, AttentionLSTM)
                input_raw = input_df.values
                input_scaled = scaler.transform(input_raw)
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    pred_change = model(input_tensor).cpu().numpy().flatten()
                
                last_val_scaled = input_scaled[-1, 0]
                pred_val_scaled = pred_change + last_val_scaled
                pred_final = scaler.inverse_transform_col(pred_val_scaled, 0)
                results["Attention LSTM"] = pred_final
            except Exception as e:
                st.error(f"Attention LSTM 예측 중 오류 발생: {e}")

        # 결과 시각화
        st.subheader(f"📊 예측 결과 분석 ({predict_date} ~ +5일)")
        
        target_dates = pd.date_range(start=predict_date, periods=HORIZON, freq='B')
        date_strs = target_dates.strftime('%Y-%m-%d')
        
        res_df = pd.DataFrame({"날짜": date_strs})
        for name, val in results.items():
            res_df[name] = np.round(val, 2)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("##### 📋 예측 가격 테이블")
            st.dataframe(res_df, hide_index=True, use_container_width=True)
            
        with col2:
            st.markdown("##### 📈 주가 추세 그래프")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            colors = {"CNN+LSTM": "#ff4b4b", "Attention LSTM": "#1c83e1"}
            styles = {"CNN+LSTM": "-", "Attention LSTM": "--"}
            
            # 예측값 Plot
            for name, val in results.items():
                ax.plot(res_df['날짜'], val, label=name, 
                        color=colors.get(name, "gray"), 
                        linestyle=styles.get(name, "-"), marker='o', linewidth=2)
            
            # 과거 데이터 (Context)
            past_days = 20
            past_data = df.loc[:cutoff_date, feature_cols[0]].tail(past_days)
            ax.plot(past_data.index.strftime('%Y-%m-%d'), past_data.values, color='gray', alpha=0.4, label='History')
            
            # 그래프 스타일링
            ax.set_title("KOSPI Forecast Trend")
            ax.set_ylabel("Index Price")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
