import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.font_manager as fm

# --- 0. 폰트 및 장치 설정 ---
try:
    import requests
    font_url = 'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf'
    font_data = requests.get(font_url).content
    font_path = 'NanumGothic-Regular.ttf'
    with open(font_path, 'wb') as f:
        f.write(font_data)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
except:
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device: {DEVICE} ---")

# --- 1. 데이터 전처리 및 로드 ---

class CustomMinMaxScaler:
    """Numpy 기반 Scaler"""
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
    
    # [수정] transform 메서드 추가
    def transform(self, data):
        if self.min_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (data - self.min_) / self.scale_
    
    def inverse_transform_col(self, data, col_index):
        return (data * self.scale_[col_index]) + self.min_[col_index]

def load_data(file_path, end_date):
    print(f"\n[Data Load] '{file_path}' 로드 중...")
    
    # 인코딩 자동 감지 로직
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'utf-16']
    df = pd.DataFrame()
    
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, sep='\t', index_col='Date', parse_dates=['Date'], encoding=enc)
            print(f"-> 성공: 인코딩 '{enc}'로 파일을 읽었습니다.")
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    if df.empty:
        print(f"오류: 파일을 읽을 수 없습니다. 인코딩이나 구분자(탭/콤마)를 확인해주세요.")
        return pd.DataFrame()

    try:
        df = df[df.index.notna()]
        df.sort_index(inplace=True)
    except Exception as e:
        print(f"데이터 전처리 실패: {e}")
        return pd.DataFrame()

    # Feature 확장
    candidate_cols = [
        'KOSPI_Close', 'KOSPI_Open', 'KOSPI_High', 'KOSPI_Low', 'KOSPI_Volume',
        'NAS_Close', 'NAS_Open', 'NAS_Volume',
        'Rate', 'VKOSPI_close', 'WTI_Close', 'USD_KRW_Close', 'KOSPI_future', 'Foreign_rate'
    ]
    
    selected_cols = []
    for col in candidate_cols:
        if col in df.columns:
            selected_cols.append(col)
    
    if not selected_cols:
        print("오류: 유효한 컬럼을 찾을 수 없습니다.")
        return pd.DataFrame()
        
    print(f"--- 선택된 Feature ({len(selected_cols)}개) ---")
    print(selected_cols)
    
    df_selected = df[selected_cols]
    
    # 날짜 필터링
    df_filtered = df_selected.loc[:end_date].copy()
    df_filtered.ffill(inplace=True)
    df_filtered.dropna(inplace=True)
    
    return df_filtered

def create_sequences(data, time_step, horizon):
    X, y = [], []
    for i in range(len(data) - time_step - horizon + 1):
        X.append(data[i:(i + time_step), :])
        # 타겟: 미래 5일 변동폭
        last_price = data[i + time_step - 1, 0] # 0번 컬럼이 KOSPI_Close라고 가정
        future_prices = data[i + time_step : i + time_step + horizon, 0]
        y.append(future_prices - last_price)
    return np.array(X), np.array(y)

# --- 2. 다양한 모델 정의 (Factory Pattern) ---

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

class CNNModel(BaseModel):
    def __init__(self, input_dim, seq_len, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
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
        x = x.permute(0, 2, 1)
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

# --- 3. 실험 및 학습 도구 ---

def train_and_evaluate(model_class, model_params, train_loader, val_loader, test_input, epochs=50):
    model = model_class(**model_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            val_loss += criterion(model(X), y).item()
    rmse = np.sqrt(val_loss / len(val_loader))
    
    with torch.no_grad():
        pred = model(test_input.to(DEVICE)).cpu().numpy().flatten()
        
    return rmse, pred

# --- 4. 메인 실행 ---

def main():
    # --- 설정 ---
    FILE_PATH = "KOSPI_base.csv"
    
    # 훈련용 데이터 Cut-off (과거 학습용)
    INPUT_CUTOFF_DATE = '2025-11-14'
    
    # 실제값 비교를 위해 로드할 최종 날짜 (CSV에 존재한다면 로드)
    # 11월 17일 ~ 21일의 실제 주가를 확인하기 위해 21일까지 로드 시도
    TARGET_START_DATE = '2025-11-17'
    TARGET_END_DATE = '2025-11-21'
    
    TIME_STEP = 60
    HORIZON = 5
    BATCH_SIZE = 32
    EPOCHS = 70
    
    # [수정] 데이터 로드 시 TARGET_END_DATE까지 로드 (실제값 추출용)
    df_full = load_data(FILE_PATH, TARGET_END_DATE)
    if df_full.empty: return

    # 모델 학습용 데이터는 INPUT_CUTOFF_DATE까지만 사용 (미래 데이터 참조 방지)
    df_train_source = df_full.loc[:INPUT_CUTOFF_DATE]

    # 스케일링 (훈련 데이터 기준으로 fit)
    scaler = CustomMinMaxScaler()
    scaled_train_data = scaler.fit_transform(df_train_source.values)
    
    # 전체 데이터 스케일링 (실제값 비교 등 편의를 위해 full 데이터도 transform만 수행)
    scaled_full_data = scaler.transform(df_full.values) if len(df_full) > len(df_train_source) else scaled_train_data
    
    # 데이터셋 생성
    X_all, y_all = create_sequences(scaled_train_data, TIME_STEP, HORIZON)
    
    # Train/Val 분리
    split = int(len(X_all) * 0.85)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=BATCH_SIZE)
    
    # 11월 17일 예측을 위한 입력 (11/14 기준 과거 60일 데이터)
    last_input_seq = scaled_train_data[-TIME_STEP:].reshape(1, TIME_STEP, -1)
    test_input = torch.tensor(last_input_seq, dtype=torch.float32)

    # --- 모델 비교 실험 ---
    print("\n========== 모델 비교 실험 시작 ==========")
    
    input_dim = df_train_source.shape[1]
    models_config = [
        ("CNN", CNNModel, {"input_dim": input_dim, "seq_len": TIME_STEP, "output_dim": HORIZON}),
        ("CNN+LSTM", CNN_LSTM, {"input_dim": input_dim, "hidden_dim": 64, "output_dim": HORIZON}),
        ("LSTM (Basic)", SimpleLSTM, {"input_dim": input_dim, "hidden_dim": 64, "output_dim": HORIZON}),
        ("LSTM (Attention)", AttentionLSTM, {"input_dim": input_dim, "hidden_dim": 64, "output_dim": HORIZON}),
        ("LSTM (Deep)", SimpleLSTM, {"input_dim": input_dim, "hidden_dim": 128, "output_dim": HORIZON})
    ]
    
    results = []
    predictions = {}
    
    last_price_scaled = scaled_train_data[-1, 0] # 11/14 기준 종가
    
    for name, cls, params in models_config:
        print(f"Training {name}...")
        start = time.time()
        rmse, pred_change = train_and_evaluate(cls, params, train_loader, val_loader, test_input, epochs=EPOCHS)
        elapsed = time.time() - start
        
        # 변동폭 -> 절대 가격 변환
        pred_price_scaled = pred_change + last_price_scaled
        pred_price_final = scaler.inverse_transform_col(pred_price_scaled, 0)
        
        results.append((name, rmse, elapsed))
        predictions[name] = pred_price_final
        print(f"  -> Val RMSE (Scaled): {rmse:.4f}, Time: {elapsed:.1f}s")

    # --- 결과 시각화 및 테이블 생성 ---
    
    # [추가] 1. 발표용 표 데이터 생성 (DataFrame)
    target_dates = pd.date_range(start=TARGET_START_DATE, periods=HORIZON, freq='B') # 영업일 기준 5일
    date_strs = target_dates.strftime('%Y-%m-%d')
    
    # 결과 테이블 초기화
    summary_df = pd.DataFrame(index=date_strs)
    
    # 실제값(Actual) 채우기 (CSV에 존재할 경우)
    actual_prices = []
    for d in date_strs:
        if d in df_full.index.strftime('%Y-%m-%d'):
            # 컬럼명이 'KOSPI_Close'라고 가정 (load_data에서 선택된 첫번째 컬럼)
            actual_val = df_full.loc[d, df_full.columns[0]] 
            actual_prices.append(actual_val)
        else:
            actual_prices.append(np.nan) # 데이터 없음
    
    summary_df['Actual (KOSPI)'] = actual_prices
    
    # 예측값(Prediction) 채우기
    for name, pred in predictions.items():
        summary_df[f'Pred ({name})'] = np.round(pred, 2)
        
    # 오차(Difference) 계산 (Basic LSTM 기준 예시)
    if 'Pred (LSTM (Basic))' in summary_df.columns:
        summary_df['Diff (Basic)'] = summary_df['Pred (LSTM (Basic))'] - summary_df['Actual (KOSPI)']

    print("\n========== [발표용] 예측 결과 테이블 ==========")
    print(summary_df)
    
    # CSV로 저장 (발표 자료 활용용)
    table_filename = 'final_prediction_table.csv'
    summary_df.to_csv(table_filename)
    print(f"\n[Save] 표 데이터 저장 완료: {table_filename}")

    # 2. 모델별 RMSE 비교 막대 그래프
    names = [r[0] for r in results]
    rmses = [r[1] for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, rmses, color=['gray', 'gray', 'skyblue', 'royalblue', 'navy'])
    plt.title("모델별 성능 비교 (RMSE 낮을수록 좋음)")
    plt.ylabel("Validation RMSE (Scaled)")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.4f}', ha='center', va='bottom')
    plt.savefig('model_comparison_result.png')
    
    # 3. 11월 17일 ~ 21일 예측 결과 비교 그래프
    plt.figure(figsize=(12, 6))
    
    # 실제값 그래프 (데이터가 있을 경우에만 그림)
    if not summary_df['Actual (KOSPI)'].isna().all():
        plt.plot(summary_df.index, summary_df['Actual (KOSPI)'], label="Actual", color='black', linewidth=3, marker='s')

    for name, pred in predictions.items():
        style = '--' if 'LSTM' in name else ':'
        width = 2 if 'Attention' in name else 1.5
        plt.plot(date_strs, pred, label=f"{name}", linestyle=style, linewidth=width, marker='o')
        
    plt.title("2025년 11월 17일 ~ 21일 KOSPI 예측 (모델별 비교)")
    plt.xlabel("날짜")
    plt.ylabel("KOSPI 지수")
    plt.legend()
    plt.grid(True)
    plt.savefig('final_forecast_comparison.png')
    print("[Save] 최종 예측 그래프 저장 완료: final_forecast_comparison.png")

if __name__ == "__main__":
    main()
