import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# [수정] 폰트 설정: 코랩 기본 폰트(DejaVu Sans) 강제 지정하여 에러 방지
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. Configuration ---
CONFIG = {
    "data_file": "KOSPI_dataset_final.csv",
    "data_start": "2013-08-06",
    "data_end": "2025-11-28",
    "train_cutoff_date": "2025-11-14", # 학습용 데이터 끝나는 날짜
    "test_start_date": "2025-11-17",   # RMSE 평가 시작 날짜
    "test_end_date": "2025-11-21",     # RMSE 평가 종료 날짜

    "plot_start_date": "2023-04-01",   # 그래프 그릴 시작 날짜

    "seq_length": 5,
    "predict_horizon": 5,

    "hidden_size": 256,
    "num_layers": 1,
    "num_classes": 1,

    "cnn_num_layers": 1,
    "num_filters": 32,
    "kernel_size": 5,

    "batch_size": 256,
    "epochs": 100, # 실험 속도를 위해 50으로 설정 (필요시 100으로 증가)
    "learning_rate": 0.005,
    "patience": 5,

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- Feature Groups Definition ---
FEATURE_GROUPS = {
    "KOSPI": ['KOSPI_Close', 'KOSPI_Open', 'KOSPI_High', 'KOSPI_Low', 'KOSPI_Volume', 'KOSPI_Amount', 'KOSPI_Change', 'KOSPI_Fluctuation', 'KOSPI_UpDown'],
    "NASDAQ": ['NAS_Open', 'NAS_High', 'NAS_Low', 'NAS_Close', 'NAS_Volume', 'NAS_Change'],
    "VKOSPI": ['VKOSPI_Close', 'VKOSPI_Change'],
    "Rate_FX": ['USD_KRW', 'EUR_KRW', 'Rate'],
    "Foreign": ['Foreign_MarketCap_Ratio', 'Foreign_MarketCap', 'Foreign_Rate'], # Foreign_Rate는 csv 컬럼명 확인 필요
    "Future": ['Future_Close', 'Future_Change'],
    "Oil": ['WTI_Close', 'WTI_Change']
}

print(f"Using Device: {CONFIG['device']}")

# --- 2. Data Processing ---
def load_data(config):
    if not os.path.exists(config["data_file"]):
        raise FileNotFoundError(f"File not found: {config['data_file']}")

    encodings_to_try = ['utf-16', 'utf-8', 'utf-8-sig', 'cp949', 'latin1']
    df = None
    for enc in encodings_to_try:
        try:
            temp_df = pd.read_csv(config["data_file"], sep='\t', index_col="Date", parse_dates=True, encoding=enc)
            if len(temp_df.columns) > 1: df = temp_df; break
            temp_df = pd.read_csv(config["data_file"], sep=',', index_col="Date", parse_dates=True, encoding=enc)
            if len(temp_df.columns) > 1: df = temp_df; break
        except: continue

    if df is None: raise ValueError("Failed to read file.")

    for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.loc[config["data_start"]:config["data_end"]]
    df = df.ffill().bfill()
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.columns = [c.strip() for c in df.columns]

    return df

def process_features(df, drop_group_name=None):
    """특정 피처 그룹을 제거하고 데이터셋을 준비함"""

    target_col = "KOSPI_Close"

    # 전체 사용 가능한 피처 식별
    available_cols = df.columns.tolist()

    # 제거할 피처 목록 결정
    cols_to_drop = []
    if drop_group_name and drop_group_name in FEATURE_GROUPS:
        cols_to_drop = FEATURE_GROUPS[drop_group_name]
        # 실제 데이터에 있는 것만 필터링
        cols_to_drop = [c for c in cols_to_drop if c in available_cols]

    # [중요] 타겟 컬럼(KOSPI_Close)은 y생성을 위해 제거하면 안 됨.
    # 하지만 X(입력)에서는 제거 실험을 할 수 있음.
    # 여기서는 y데이터를 먼저 뽑고, X데이터프레임에서 제거하는 방식을 사용.

    # 1. Target Data 추출 (y)
    raw_y = df[[target_col]].values

    # 2. Input Features 구성 (X)
    # drop_group에 target_col이 포함되어 있어도 X에서는 제거, y는 유지
    input_df = df.drop(columns=cols_to_drop, errors='ignore')

    # Scaler
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_x = scaler_x.fit_transform(input_df)
    scaled_y = scaler_y.fit_transform(raw_y)

    # Sequence Generation
    X, y = [], []
    seq_len = CONFIG["seq_length"]

    for i in range(len(scaled_x) - seq_len):
        X.append(scaled_x[i : i + seq_len])
        y.append(scaled_y[i + seq_len, 0]) # Next step prediction

    X = np.array(X)
    y = np.array(y)

    # Dates alignment
    dates = df.index[seq_len:]

    return X, y, dates, scaler_y, len(input_df.columns)

def split_data(X, y, dates, config):
    # Split based on date
    train_end = pd.Timestamp(config["train_cutoff_date"])

    train_mask = dates <= train_end
    # Test data: Evaluation period (Nov 24 ~ Nov 28)
    test_start = pd.Timestamp(config["test_start_date"])
    test_end = pd.Timestamp(config["test_end_date"])
    test_mask = (dates >= test_start) & (dates <= test_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_dates = dates[test_mask]

    # For long-term plotting
    plot_start = pd.Timestamp(config["plot_start_date"])
    plot_mask = dates >= plot_start
    X_plot = X[plot_mask]
    y_plot = y[plot_mask]
    plot_dates = dates[plot_mask]

    # To Tensor
    device = config['device']
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    X_plot_t = torch.FloatTensor(X_plot).to(device)

    return (X_train_t, y_train_t), (X_test_t, y_test_t, test_dates), (X_plot_t, y_plot, plot_dates)

# --- 3. Models ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, num_filters, kernel_size, seq_length):
        super().__init__()
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
        super().__init__()
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

# --- 4. Training ---
def train_model(model, train_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    model.train()

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(config["epochs"]):
        epoch_loss = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            if torch.isnan(loss): return model # Fail safe
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]: break
    return model

# --- 5. Main Logic ---
def main():
    print("Loading Data...")
    full_df = load_data(CONFIG)
    print(f"Data range: {full_df.index.min()} ~ {full_df.index.max()}")

    # 1. Ablation Study Scenarios
    scenarios = ["Base (All Features)"] + [f"Remove {g}" for g in FEATURE_GROUPS.keys()]

    results = [] # Store RMSE results
    best_rmse = float('inf')
    best_scenario = "Base (All Features)"

    # Store long-term predictions for plotting (only for base or best)
    long_term_preds = {}

    print("\n[Start Ablation Study]")

    for scenario in scenarios:
        drop_group = scenario.replace("Remove ", "") if "Remove" in scenario else None
        print(f"\n>> Experiment: {scenario}")

        # Prepare Data
        X, y, dates, scaler_y, input_dim = process_features(full_df, drop_group)
        train_data, test_data, plot_data = split_data(X, y, dates, CONFIG)

        train_loader = DataLoader(TensorDataset(*train_data), batch_size=CONFIG["batch_size"], shuffle=True)

        # Define Models [수정: LSTM Attention 추가]
        models = {
            "CNN": CNNModel(input_dim, 1, 32, 5, CONFIG["seq_length"]),
            "LSTM": LSTMModel(input_dim, 256, 1, 1),
            "CNN+LSTM": CNNLSTMModel(input_dim, 256, 1, 1, 32, 5),
            "LSTM(Attn)": LSTMAttentionModel(input_dim, 256, 1, 1)
        }

        scenario_rmse = {}

        for name, model in models.items():
            model.to(CONFIG['device'])
            train_model(model, train_loader, CONFIG)

            # Evaluate on Test (Nov 24 ~ Nov 28)
            model.eval()
            with torch.no_grad():
                pred_scaled = model(test_data[0]).cpu().numpy()
                y_true_scaled = test_data[1].cpu().numpy()

            pred_inv = scaler_y.inverse_transform(pred_scaled)
            y_true_inv = scaler_y.inverse_transform(y_true_scaled)

            rmse = np.sqrt(mean_squared_error(y_true_inv, pred_inv))
            scenario_rmse[name] = rmse
            print(f"   [{name}] RMSE: {rmse:.4f}")

            # Save predictions for Base scenario (to plot long term graph)
            if scenario == "Base (All Features)":
                model.eval()
                with torch.no_grad():
                    plot_pred_scaled = model(plot_data[0]).cpu().numpy()
                plot_pred_inv = scaler_y.inverse_transform(plot_pred_scaled)
                long_term_preds[name] = pd.Series(plot_pred_inv.flatten(), index=plot_data[2])

                # Save Actuals once
                if "Actual" not in long_term_preds:
                    y_plot_inv = scaler_y.inverse_transform(plot_data[1].reshape(-1,1))
                    long_term_preds["Actual"] = pd.Series(y_plot_inv.flatten(), index=plot_data[2])

        # Record Results
        row = {"Scenario": scenario}
        row.update(scenario_rmse)
        results.append(row)

    # --- 6. Results & Plotting ---
    print("\n[Ablation Study Results]")
    df_res = pd.DataFrame(results)
    print(df_res)
    df_res.to_csv("ablation_study_results.csv", index=False)

    # 2. Long-term Plotting (2023-04 ~ 2025-11-28)
    print("\n[Drawing Long-term Comparison Plot...]")
    plt.figure(figsize=(14, 7))

    # Actual
    plt.plot(long_term_preds["Actual"].index, long_term_preds["Actual"], label='Actual (KOSPI)', color='black', alpha=0.6, linewidth=1.5)

    # Models [수정: LSTM(Attn) 색상 추가]
    colors = {'CNN': 'green', 'LSTM': 'blue', 'CNN+LSTM': 'red', 'LSTM(Attn)': 'purple'}
    for name, pred_series in long_term_preds.items():
        if name == "Actual": continue
        plt.plot(pred_series.index, pred_series, label=name, color=colors.get(name, 'orange'), alpha=0.8, linewidth=1)

    plt.title(f"Long-term KOSPI Prediction ({CONFIG['plot_start_date']} ~ {CONFIG['data_end']})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("long_term_prediction_comparison.png", dpi=300)
    print("Graph Saved: long_term_prediction_comparison.png")

if __name__ == "__main__":
    main()