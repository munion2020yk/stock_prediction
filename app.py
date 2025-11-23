import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import io
import time

# --- 1. 설정 및 클래스 정의 ---

st.set_page_config(page_title="Advanced Stock Prediction App", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scaler 클래스
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
    
    def transform(self, data):
        if self.min_ is None:
            raise RuntimeError("Scaler not fitted.")
        return (data - self.min_) / self.scale_
    
    def inverse_transform_col(self, data, col_index):
        return (data * self.scale_[col_index]) + self.min_[col_index]

# --- 모델 클래스 정의 (Factory Pattern 활용을 위해 통일된 부모 클래스 사용 가능하지만, 여기선 개별 정의) ---

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

class CNNModel(BaseModel):
    def __init__(self, input_dim, seq_len, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        # CNN 출력 크기 계산: pool(L) -> L//2
        self.fc1 = nn.Linear(64 * (seq_len // 2), 50)
        self.fc2 = nn.Linear(50, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1) # (N, C, L)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class SimpleLSTM(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.relu(h_n[-1]))

class CNN_LSTM(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv(x)))
        x = x.permute(0, 2, 1) # LSTM 입력을 위해 다시 (N, L, C)로 변환
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class AttentionLSTM(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        score = self.attention(outputs)
        weights = F.softmax(score, dim=1)
        context = torch.sum(outputs * weights, dim=1)
        return self.fc(context)

# 데이터 로드 함수
@st.cache_data
def load_data(uploaded_file):
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'utf-16']
    df = pd.DataFrame()
    bytes_data = uploaded_file.getvalue()
    
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(bytes_data), sep='\t', index_col='Date', parse_dates=['Date'], encoding=enc)
            if len(df.columns) <= 1:
                df = pd.read_csv(io.BytesIO(bytes_data), sep=',', index_col='Date', parse_dates=['Date'], encoding=enc)
            break
        except:
            continue
            
    if not df.empty:
        df = df[df.index.notna()]
        df.sort_index(inplace=True)
        df.ffill(inplace=True)
        df.dropna(inplace=True)
    return df

# 시퀀스 생성 함수
def create_sequences(data, time_step, horizon):
    X, y = [], []
    for i in range(len(data) - time_step - horizon + 1):
        X.append(data[i:(i + time_step), :])
        last_price = data[i + time_step - 1, 0] # 0번 컬럼 = Target
        future_prices = data[i + time_step : i + time_step + horizon, 0]
        y.append(future_prices - last_price)
    return np.array(X), np.array(y)

# 학습 함수 (개별 모델용)
def train_model(model, train_loader, epochs, progress_bar_slot, model_idx, total_models):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    progress_bar = progress_bar_slot.progress(0, text=f"[{model_idx}/{total_models}] Training...")
    
    for epoch in range(epochs):
        avg_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        
        # 진행률 업데이트 (너무 자주는 생략 가능)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            progress_bar.progress((epoch + 1) / epochs, text=f"[{model_idx}/{total_models}] Epoch {epoch+1}/{epochs} (Loss: {avg_loss/len(train_loader):.5f})")
    
    return model

# --- 2. Streamlit UI 구성 ---

def main():
    st.title("📈 Advanced Stock Prediction: Multi-Model Ensemble")
    st.markdown("""
    KOSPI 데이터를 업로드하면 **5가지 다른 모델**로 학습하고, 
    그 결과를 종합한 **앙상블(평균)** 예측값을 제공합니다.
    """)

    # 사이드바: 설정
    st.sidebar.header("1. 데이터 및 설정")
    uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (KOSPI_base.csv)", type=['csv', 'txt'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df.empty:
            st.error("파일을 읽을 수 없습니다.")
            return

        st.sidebar.success("데이터 로드 성공!")
        
        # 2. Feature 선택
        st.sidebar.subheader("2. Feature 선택")
        all_columns = df.columns.tolist()
        target_col = 'KOSPI_Close'
        if target_col not in all_columns:
            st.error(f"'{target_col}' 컬럼이 없습니다.")
            return

        feature_options = [c for c in all_columns if c != target_col]
        selected_features = st.sidebar.multiselect(
            "보조 지표 선택",
            options=feature_options,
            default=feature_options[:min(4, len(feature_options))]
        )
        final_cols = [target_col] + selected_features
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. 예측 설정")
        
        min_date = df.index.min().date()
        max_date = df.index.max().date()
        default_pred_date = max_date + pd.Timedelta(days=1)
        if default_pred_date.weekday() >= 5:
             default_pred_date += pd.Timedelta(days=(7 - default_pred_date.weekday()))
             
        predict_start_date = st.sidebar.date_input(
            "예측 시작 날짜",
            value=default_pred_date,
            min_value=min_date + pd.Timedelta(days=60)
        )
        
        epochs = st.sidebar.slider("모델별 학습 Epochs", 10, 100, 30) # 다중 모델이므로 기본값 낮춤
        time_step = 60
        horizon = 5

        st.subheader("📊 데이터 미리보기")
        st.write(f"**Target:** {target_col} | **Features:** {selected_features}")
        st.dataframe(df[final_cols].tail(5))

        if st.button("🚀 전체 모델 학습 및 예측 시작", type="primary"):
            
            # --- 데이터 준비 ---
            cutoff_date = pd.to_datetime(predict_start_date) - pd.Timedelta(days=1)
            train_df = df.loc[:cutoff_date, final_cols]
            
            if len(train_df) < time_step + horizon:
                st.error("데이터 부족.")
                return

            st.info(f"학습 기간: {train_df.index.min().date()} ~ {train_df.index.max().date()}")

            scaler = CustomMinMaxScaler()
            train_data = train_df.values
            scaled_train_data = scaler.fit_transform(train_data)
            
            X, y = create_sequences(scaled_train_data, time_step, horizon)
            
            batch_size = 32
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # --- 모델 구성 ---
            input_dim = len(final_cols)
            
            # (모델명, 클래스, 파라미터)
            models_config = [
                ("CNN", CNNModel, {"input_dim": input_dim, "seq_len": time_step, "output_dim": horizon}),
                ("CNN+LSTM", CNN_LSTM, {"input_dim": input_dim, "hidden_dim": 64, "output_dim": horizon}),
                ("LSTM(Basic)", SimpleLSTM, {"input_dim": input_dim, "hidden_dim": 64, "output_dim": horizon}),
                ("LSTM(Attention)", AttentionLSTM, {"input_dim": input_dim, "hidden_dim": 64, "output_dim": horizon}),
                ("LSTM(Deep)", SimpleLSTM, {"input_dim": input_dim, "hidden_dim": 128, "output_dim": horizon})
            ]
            
            predictions = {} # 결과 저장용
            
            # 예측용 입력 데이터 (마지막 60일)
            last_sequence = scaled_train_data[-time_step:] 
            input_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            last_close_scaled = last_sequence[-1, 0]

            # --- 학습 루프 ---
            total_models = len(models_config)
            cols = st.columns(total_models) # 진행바를 가로로 배치하거나
            
            main_progress = st.container() # 메인 진행 영역
            
            for idx, (name, cls, params) in enumerate(models_config):
                with main_progress:
                    st.write(f"**{idx+1}. {name} 모델 학습 중...**")
                    prog_bar = st.empty()
                    
                    # 모델 초기화
                    model = cls(**params)
                    
                    # 학습
                    model = train_model(model, train_loader, epochs, prog_bar, idx+1, total_models)
                    
                    # 예측
                    model.eval()
                    with torch.no_grad():
                        pred_change = model(input_tensor).cpu().numpy().flatten()
                    
                    # 스케일 복원
                    pred_price_scaled = pred_change + last_close_scaled
                    pred_price = scaler.inverse_transform_col(pred_price_scaled, 0)
                    
                    predictions[name] = pred_price
                    st.toast(f"{name} 학습 완료!")

            st.success("모든 모델 학습 완료!")
            
            # --- 결과 집계 및 시각화 ---
            st.markdown("---")
            st.subheader(f"📅 예측 결과 비교 ({predict_start_date} ~)")
            
            pred_dates = pd.date_range(start=predict_start_date, periods=horizon, freq='B')
            
            # DataFrame 생성
            res_data = {"날짜": pred_dates.strftime('%Y-%m-%d')}
            
            # 각 모델별 예측값 추가
            ensemble_preds = np.zeros(horizon)
            for name, pred in predictions.items():
                res_data[name] = np.round(pred, 2)
                ensemble_preds += pred
            
            # 앙상블 (평균) 계산
            ensemble_preds /= total_models
            res_data["Ensemble (Avg)"] = np.round(ensemble_preds, 2)
            
            # 실제값 (데이터가 있다면)
            full_df = df.loc[predict_start_date:, target_col]
            actual_vals = []
            for d in pred_dates:
                actual_vals.append(full_df.loc[d] if d in full_df.index else None)
            
            if any(v is not None for v in actual_vals):
                res_data["Actual"] = actual_vals

            result_df = pd.DataFrame(res_data)
            
            # 테이블 표시 (앙상블 강조)
            st.dataframe(result_df.style.highlight_max(axis=0, color='lightgreen', subset=["Ensemble (Avg)"]), use_container_width=True)
            
            # 그래프
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 개별 모델 (점선, 얇게)
            for name in predictions.keys():
                ax.plot(result_df['날짜'], result_df[name], linestyle=':', alpha=0.7, label=name)
            
            # 앙상블 (실선, 굵게, 빨강)
            ax.plot(result_df['날짜'], result_df["Ensemble (Avg)"], color='red', linewidth=3, marker='o', label='Ensemble (Avg)')
            
            # 실제값 (검정 실선)
            if "Actual" in result_df.columns and result_df["Actual"].notna().any():
                ax.plot(result_df['날짜'], result_df["Actual"], color='black', linewidth=2, marker='s', label='Actual')

            ax.set_title("Multi-Model Prediction Comparison")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    else:
        st.info("왼쪽 사이드바에서 CSV 파일을 업로드해주세요.")

if __name__ == "__main__":
    main()
