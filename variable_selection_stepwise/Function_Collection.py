# Function_Collection.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import FinanceDataReader as fdr
import pandas as pd
from copy import deepcopy
import os
import time

from Model import LSTM, CNN_LSTM, CNNModel

pd.set_option('display.max_columns', None)

class function_collection():
    
    def __init__(self):
        self.raw_df = None
        self.results_cache = {}
        self.stepwise_log = []
        self.save_time = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # ======================================================
    # 데이터 준비 함수 (기존과 동일)
    # ======================================================
    
    def create_sequences(self, X_data, y_data, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X_data) - seq_length):
            X_seq.append(X_data[i:i+seq_length])
            y_seq.append(y_data[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
        
        
    
    def make_df(self, start, end):
        # KOSPI 기본 데이터
        df = fdr.DataReader('KS11', start, end)


        # ==========================================================================
        # VKOSPI 데이터 추가
        df_usdkrw = fdr.DataReader('USD/KRW', start, end)[['Close']].rename(columns={'Close':'USD_KRW'})
        df_eurkrw = fdr.DataReader('EUR/KRW', start, end)[['Close']].rename(columns={'Close':'EUR_KRW'})
        df_nasdaq = fdr.DataReader('IXIC', start, end)[['Close']].rename(columns={'Close':'NASDAQ'})

        # VKOSPI CSV 파일 로드
        df_vkospi = pd.read_csv("./Data/KOSPI Volatility.csv")
        df_vkospi = df_vkospi.rename(columns={'날짜':'Date', 
                                              '종가':'VKOSPI_Close',
                                              '시가':'VKOSPI_Open',
                                              '고가':'VKOSPI_High',
                                              '저가':'VKOSPI_Low',
                                              '변동 %':'VKOSPI_Change'})

        # 원하는 컬럼만 선택 (예: 종가 + 변동률만)
        df_vkospi = df_vkospi[['Date', 'VKOSPI_Close', 'VKOSPI_Open', 'VKOSPI_High', 'VKOSPI_Low','VKOSPI_Change']]
        df_vkospi['VKOSPI_Change'] = (df_vkospi['VKOSPI_Change'].str.replace('%', '', regex=False).astype(float))

        # 날짜 형식 통일
        df_vkospi['Date'] = pd.to_datetime(df_vkospi['Date'])
        df_vkospi = df_vkospi.set_index('Date').sort_index()
        
        
        # # ==========================================================================
        # # 금리 데이터 추가
        # df_market_interest_rate = pd.read_csv("./Data/market interest rate.csv", encoding='EUC-KR')
        # df_market_interest_rate = df_market_interest_rate.rename(columns={'국고채3년(평균)':'Treasury_Bond_3years',
        #                                                                   '국고채5년(평균)':'Treasury_Bond_5years',
        #                                                                   '국고채10년(평균)':'Treasury_Bond_10years',
        #                                                                   '기준금리':'Benchmark_Interest_Rate'})
        
        # # 날짜형 변환 및 인덱스 설정
        # df_market_interest_rate['Date'] = pd.to_datetime(df_market_interest_rate['Date'])
        # df_market_interest_rate = df_market_interest_rate.set_index('Date').sort_index()

        # df_market_interest_rate = df_market_interest_rate.loc[start:end, ['Treasury_Bond_3years', 'Treasury_Bond_5years', 'Treasury_Bond_10years', 'Benchmark_Interest_Rate']]
        
        
        # ==========================================================================
        # 선물 데이터 추가 
        df_future = pd.read_csv("./Data/KOSPI_Future.csv", encoding="EUC-KR")
        df_future = df_future.drop(columns=['거래량'])
        df_future = df_future.rename(columns={
                                        '날짜':'Date', 
                                        '종가':'KOSPI_Future_Close',
                                        '시가':'KOSPI_Future_Open',
                                        '고가':'KOSPI_Future_High',
                                        '저가':'KOSPI_Future_Low',
                                        '변동 %':'KOSPI_Future_Change'})
        
        # 날짜형 변환 및 인덱스 설정
        # df_future['Date'] = pd.to_datetime(df_future['Date'])
        def parse_date(x):
            x = str(x)
            if '/' in x:
                # "11/21/2025" 같은 형식
                return pd.to_datetime(x, format='%m/%d/%Y')
            else:
                # "2025-11-21" 같은 형식
                return pd.to_datetime(x, format='%Y-%m-%d')

        df_future['Date'] = df_future['Date'].apply(parse_date)
        df_future = df_future.set_index('Date').sort_index()
        df_future = df_future.loc[start:end]
        
        
        # ==========================================================================
        # WTI 유가 데이터 추가 
        df_WTI = pd.read_csv("./Data/WTI_Oil.csv", encoding="EUC-KR")
        df_WTI = df_WTI.drop(columns=['거래량'])
        df_WTI = df_WTI.rename(columns={
                                        '날짜':'Date', 
                                        '종가':'WTI_Close',
                                        '시가':'WTI_Open',
                                        '고가':'WTI_High',
                                        '저가':'WTI_Low',
                                        '변동 %':'WTI_Change'})
        
        # 날짜형 변환 및 인덱스 설정
        df_WTI['Date'] = pd.to_datetime(df_WTI['Date'])
        df_WTI = df_WTI.set_index('Date').sort_index()
        df_WTI = df_WTI.loc[start:end]
        
        
        # ==========================================================================
        # 외국인 보유량 데이터 추가
        df_Foreign = pd.read_csv("./Data/Foreign Holdings.csv", encoding="EUC-KR")
        df_Foreign = df_Foreign.drop(columns=['시가총액_전체','시가총액_외국인보유', '주식수_전체', '주식수_외국인보유', '주식수_비율'])
        df_Foreign = df_Foreign.rename(columns={'시가총액_비율':'Foreign_Holdings_ratio'})
        
        # 날짜형 변환 및 인덱스 설정
        df_Foreign['Date'] = pd.to_datetime(df_Foreign['Date'])
        df_Foreign = df_Foreign.set_index('Date').sort_index()
        df_Foreign = df_Foreign.loc[start:end]


        # ==========================================================================
        # 날짜 기준 병합
        # print(df.shape)
        # print(df_usdkrw.shape)
        # print(df_eurkrw.shape)
        # print(df_nasdaq.shape)
        # print(df_vkospi.shape)
        # print(df_market_interest_rate.shape)
        # print(df_future.shape)
        # print(df_WTI.shape)
        # print(df_Foreign.shape)
        
        df = df.join([df_usdkrw, df_eurkrw, df_nasdaq, df_vkospi, df_future, df_WTI, df_Foreign], how='left')

        for column in df.columns:
            df[column] = (
                df[column].astype(str).str.replace(',', '', regex=False).str.replace('%', '', regex=False).astype(float)
            )

        # 결측치 처리
        df = df.fillna(method='ffill').dropna()
        
        # print(df.shape)
        
        return df
    
    
    def df_to_Xy(self, df, use_columns):
        df_copy = df.copy()
        df_copy = df_copy[use_columns]
        X = df_copy.drop('Close', axis=1)
        y = df_copy[['Close']]
        return df_copy, X, y
        
        
    def load_data(self, df_index, X, y, seq_length=60, train_ratio=0.7, val_ratio=0.1, test_start_date=None):

        # ⑥ 스케일링 및 시퀀스 생성 (기존 동일)
        ss = StandardScaler()
        ms = MinMaxScaler()
        X_ss = ss.fit_transform(X)
        y_ms = ms.fit_transform(y)

        X_seq, y_seq = self.create_sequences(X_ss, y_ms, seq_length)

        # ⑦ Train / Val / Test 분리
        total_len = len(X_seq)
        
        # ⭐️ 분할 인덱스 초기화
        split_point_found = False
        train_val_end_idx = 0
        
        if test_start_date:
            test_start_date = pd.to_datetime(test_start_date)
            
            # 1. 원본 DF 인덱스에서 Test 시작 날짜에 해당하는 첫 번째 인덱스 찾기
            # (try-except 제거)
            # 조건에 맞는 첫 번째 날짜를 찾고, 그 날짜의 DF 내 위치(index location)를 찾음.
            
            # 주의: .iloc[0]을 사용하려면, 날짜가 데이터프레임 내에 존재해야 합니다.
            # 존재하지 않는 경우 에러를 발생시키는 대신, 마지막 인덱스보다 크게 설정하는 등의 처리 필요.
            
            temp_index = df_index[df_index >= test_start_date]

            if len(temp_index) > 0: # 데이터가 존재하는지 확인
                original_test_start_date = temp_index[0]
                idx_in_original_df = df_index.get_loc(original_test_start_date)
            
                if idx_in_original_df >= seq_length:
                    # 2. 시퀀스 배열(X_seq)에서의 분할 인덱스 계산
                    train_val_end_idx = idx_in_original_df - seq_length
                    split_point_found = True
                else:
                    print(f"[Warning] Test start date is too early (index: {idx_in_original_df} < seq_length: {seq_length}). Using ratio split.")
            else:
                 print(f"[Warning] Test start date '{test_start_date.strftime('%Y-%m-%d')}' is outside the data range. Using ratio split.")


        if split_point_found:
            # ⭐️ 날짜 기반 분할 인덱스가 유효할 때 (Test 기간 확정)
            
            train_val_len = min(train_val_end_idx, total_len)
            ratio_sum = train_ratio + val_ratio
            
            # Train/Val 비율은 기존 비율을 유지하며 전체 Train/Val 섹션을 나눕니다.
            train_size = int(train_val_len * (train_ratio / ratio_sum)) if ratio_sum > 0 else 0
            val_size = train_val_len - train_size
            
        else:
            # ⭐️ 비율 기반 분할 로직 (test_start_date가 없거나, 날짜 기반 분할 실패 시)
            
            train_size = int(total_len * train_ratio)
            val_size = int(total_len * val_ratio)


        # 최종 분할 적용
        X_train = X_seq[:train_size]
        y_train = y_seq[:train_size]

        X_val = X_seq[train_size:train_size+val_size]
        y_val = y_seq[train_size:train_size+val_size]

        X_test = X_seq[train_size+val_size:]
        y_test = y_seq[train_size+val_size:]

        # ⑧ Tensor 변환 (기존과 동일)
        X_train_tensors = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_val_tensors = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        X_test_tensors = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train_tensors = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        y_val_tensors = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        y_test_tensors = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        return X_train_tensors, X_val_tensors, X_test_tensors, \
            y_train_tensors, y_val_tensors, y_test_tensors, ss, ms, train_size, val_size
    


    # ======================================================
    # 학습/평가 함수 
    # ======================================================
    
    def train_model(self, model, train_loader, val_loader, epochs, lr, patience):
        """ ⭐️ [수정] OOM 방지용 미니배치 + Early Stopping 로직 추가 """
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        
        train_losses, val_losses = [], []

        # ⭐️ Early Stopping을 위한 변수 초기화
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(1, epochs+1):
            
            # === [Train Step] ===
            model.train()
            epoch_train_loss = 0.0
            
            # ⭐️ [수정] 미니배치 루프
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            train_losses.append(epoch_train_loss / len(train_loader))

            # === [Validation Step] ===
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                # ⭐️ [수정] Validation도 배치 단위로
                for X_val_batch, y_val_batch in val_loader:
                    val_outputs = model(X_val_batch)
                    val_loss = criterion(val_outputs, y_val_batch)
                    epoch_val_loss += val_loss.item()
            
            current_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(current_val_loss)

            # === [로그 출력] ===
            if epoch % 10 == 0 or epoch == 1:
                print(f"[Epoch {epoch}/{epochs}] Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

            # ⭐️ === [Early Stopping 체크] ===
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_no_improve = 0
                # 가장 좋은 모델의 가중치를 CPU 메모리에 저장
                best_model_state = deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                break # 학습 루프 탈출
        
        # ⭐️ [수정] 학습이 끝난 후, 가장 좋았던 모델 가중치를 다시 불러옴
        if best_model_state:
            print("Loading best model weights...")
            model.load_state_dict(best_model_state)

        return train_losses, val_losses
    
    def evaluate_model(self, model, X_test, y_test, scaler):
        model.eval()
        with torch.no_grad():
            # ⭐️ [수정] .cpu() 추가 (GPU->CPU)
            preds = model(X_test).detach().cpu().numpy()
            actual = y_test.detach().cpu().numpy()
            
        preds_inv = scaler.inverse_transform(preds)
        actual_inv = scaler.inverse_transform(actual)
        mse = mean_squared_error(actual_inv, preds_inv)
        rmse = np.sqrt(mse)
        return preds_inv, actual_inv, rmse

    # ======================================================
    # 실제 학습/캐싱 함수
    # ======================================================
    
    def _train_and_eval(self, columns, current_params, set_seeds_func):
        """실제 학습을 수행하는 내부 함수 (미니배치 적용)"""
        
        set_seeds_func(100)
        df, X, y = self.df_to_Xy(self.raw_df, columns)
        
        X_train, X_val, X_test, y_train, y_val, y_test, ss, ms, train_size, val_size = self.load_data(
            df.index, X, y,
            seq_length=current_params['seq_length'],
            train_ratio=current_params['train_ratio'],
            val_ratio=current_params['val_ratio'],
            test_start_date=current_params.get('test_start_date')
        )

        input_size = X_train.shape[2]
        device = self.device

        if current_params['use_LSTM']:
            model = LSTM(current_params['num_classes'], 
                         input_size, 
                         current_params['hidden_size'],
                         current_params['num_layers'],
                         current_params['seq_length']).to(device)
        
        elif current_params['use_CNN']:
            model = CNNModel(current_params['num_classes'], 
                             input_size, 
                             current_params['hidden_size'],
                             current_params['num_layers'],
                             current_params['seq_length'],
                             cnn_num_layers=current_params['cnn_num_layers'],
                             num_filters=current_params['num_filters'],
                             kernel_size=current_params['kernel_size']).to(device)
        
        else:
            model = CNN_LSTM(current_params['num_classes'], 
                             input_size, 
                             current_params['hidden_size'],
                             current_params['num_layers'],
                             current_params['seq_length'],
                             cnn_num_layers=current_params['cnn_num_layers'],
                             num_filters=current_params['num_filters'],
                             kernel_size=current_params['kernel_size']).to(device)

        # ==================================
        # ⭐️ DataLoader 생성
        # ==================================
        BATCH_SIZE = current_params['batch_size']
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # ==================================
        
        start = time.time()
        
        # ⭐️ train_model에 loader 전달
        train_losses, val_losses = self.train_model(
            model, train_loader, val_loader, 
            current_params['epochs'], 
            current_params['learning_rate'],
            current_params['patience']
        )

        end = time.time()
        train_time = float(f"{end - start:.3f}")
        
        self.save_time.append(train_time)
        
        print(f"\n✅ Training Time: {train_time} sec")

        preds_inv, actual_inv, rmse = self.evaluate_model(model, X_test, y_test, ms)
        rmse = round(rmse, 4)
        print(f"✅ Test RMSE: {rmse}")

        return {
            "df": df, "rmse": rmse, 
            # ⭐️ OOM 방지를 위해 모델을 CPU로 이동시켜 저장
            "model": deepcopy(model.cpu()), 
            "features" : columns,
            "train_losses": train_losses, "val_losses": val_losses,
            "preds_inv": preds_inv, "actual_inv": actual_inv,
            "train_size": train_size, "val_size": val_size,
            "params_used": deepcopy(current_params)
        }

    def train_and_eval(self, columns, current_params, set_seeds_func):
        """캐시를 확인하는 래퍼 함수 (main.py에서 이동)"""
        
        # ⭐️ [수정] 캐시 '키'에 batch_size 추가
        params_tuple = (
            current_params['batch_size'],
            current_params['seq_length'],
            current_params['hidden_size'],
            current_params['num_layers'],
            current_params['cnn_num_layers'] if not current_params['use_LSTM'] else -1,
            current_params['num_filters'] if not current_params['use_LSTM'] else -1,
            current_params['kernel_size'] if not current_params['use_LSTM'] else -1,
        )
        cache_key = (tuple(sorted(columns)), params_tuple)
        
        if cache_key in self.results_cache:
            print(f"\n[CACHE] Using cached result for {columns}")
            cached_result = self.results_cache[cache_key]
            self.save_time.append(0.000)
            print(f"✅ (Cached) Test RMSE: {cached_result['rmse']}")
            return cached_result
        else:
            print(f"\n[TRAIN] Training new combination {columns}")
            new_result = self._train_and_eval(columns, current_params, set_seeds_func)
            self.results_cache[cache_key] = new_result
            return new_result

    # ======================================================
    # Stepwise 로직 함수 (기존과 동일)
    # ======================================================
    def run_stepwise_selection(self, current_params, model_name, set_seeds_func):
        # ( ... 기존 run_stepwise_selection 로직과 동일 ... )
        # ( ... 이 함수는 각 Grid의 "1등"을 반환 ... )
        
        BASE_FEATURES = current_params['BASE_FEATURES']
        CANDIDATES = current_params['CANDIDATES']
        MIN_IMPROVE = 0.001
        MAX_FEATURES = 15
        
        best_of_all_runs = {"rmse": float("inf"), "model": None, "features": None}
        
        starting_points = []
        starting_points.append( (deepcopy(BASE_FEATURES), deepcopy(CANDIDATES)) )
        for cand in CANDIDATES:
            new_start_features = deepcopy(BASE_FEATURES) + [cand]
            new_candidates = [c for c in CANDIDATES if c != cand]
            starting_points.append( (new_start_features, new_candidates) )

        print(f"\n🔥 총 {len(starting_points)}개의 다른 시작점에서 Stepwise 탐색을 시작합니다.")

        total_runs = len(starting_points)
        for run_count, (start_features, start_candidates) in enumerate(starting_points, 1):
            
            print(f"\n\n{'='*60}")
            print(f"🚀 [Run {run_count}/{total_runs}] Start Features: {start_features}")
            print(f"{'='*60}")
            
            selected = deepcopy(start_features)
            candidates = deepcopy(start_candidates)
            best_for_this_run = {"rmse": float("inf"), "model": None, "features": None}

            print(f"\n[Stepwise 시작] 초기 변수:\n{selected}\n")
            result = self.train_and_eval(selected, current_params, set_seeds_func)
            best_rmse = result["rmse"]
            print(f"✅ 초기 RMSE = {best_rmse:.6f}")
            # self.stepwise_log.append(f"초기 RMSE: {best_rmse:.6f}")
            # self.stepwise_log.append(f"        (Features: {selected})\n")
            
            best_for_this_run = deepcopy(result)
            
            while True:
                improved = False
                print("\n📌 [Forward Step]")
                best_forward_var = None
                best_forward_rmse = best_rmse

                for var in candidates:
                    trial_cols = selected + [var]
                    result = self.train_and_eval(trial_cols, current_params, set_seeds_func)
                    rmse = result["rmse"]
                    # self.stepwise_log.append(f"        {self.save_time[-1]:.3f} sec")
                    # self.stepwise_log.append(f"        [Forward] + {var:<10} → RMSE = {rmse:.6f}")
                    # self.stepwise_log.append(f"        {trial_cols}\n")
                    if rmse < best_for_this_run["rmse"]:
                        best_for_this_run = deepcopy(result)
                    if rmse < best_forward_rmse:
                        best_forward_var = var
                        best_forward_rmse = rmse

                if best_forward_var is not None and (best_rmse - best_forward_rmse) > MIN_IMPROVE:
                    selected.append(best_forward_var)
                    candidates.remove(best_forward_var)
                    best_rmse = best_forward_rmse
                    improved = True
                    print(f"★ Forward 선택: {best_forward_var}  → RMSE = {best_rmse:.6f}")
                else:
                    print("\nForward 개선 없음")

                if len(selected) >= MAX_FEATURES:
                    print("최대 변수 개수 도달 → 종료")
                    break

                print("\n📌 [Backward Step]")
                removable = [v for v in selected if v not in BASE_FEATURES]
                best_backward_var = None
                best_backward_rmse = best_rmse

                for var in removable:
                    trial_cols = [v for v in selected if v != var]
                    result = self.train_and_eval(trial_cols, current_params, set_seeds_func)
                    rmse = result["rmse"]
                    # self.stepwise_log.append(f"        {self.save_time[-1]:.3f} sec")
                    # self.stepwise_log.append(f"        [Backward] - {var:<10} → RMSE = {rmse:.6f}")
                    # self.stepwise_log.append(f"        {trial_cols}\n")
                    if rmse < best_for_this_run["rmse"]:
                        best_for_this_run = deepcopy(result)
                    if rmse < best_backward_rmse:
                        best_backward_var = var
                        best_backward_rmse = rmse

                if best_backward_var is not None and (best_rmse - best_backward_rmse) > MIN_IMPROVE:
                    selected.remove(best_backward_var)
                    candidates.append(best_backward_var)
                    best_rmse = best_backward_rmse
                    improved = True
                    print(f"★ Backward 제거: {best_backward_var} → RMSE = {best_rmse:.6f}")
                else:
                    print("Backward 없음")

                if not improved:
                    print("\n🚫 개선 없음 → Stepwise 종료")
                    break
            
            print(f"\n[Run {run_count} 종료] 이 경로의 최적 RMSE: {best_for_this_run['rmse']:.6f}")

            if best_for_this_run["rmse"] < best_of_all_runs["rmse"]:
                print(f"🎉 [Grid Best 갱신] 신규 RMSE: {best_for_this_run['rmse']:.6f}")
                best_of_all_runs = deepcopy(best_for_this_run)
        
        return best_of_all_runs

    # ======================================================
    # ⭐️ 전체 실험 컨트롤 타워 (저장 로직 이동)
    # ======================================================
    def run_grid_search_experiment(self, hyper_parameters, set_seeds_func):
        
        if hyper_parameters['use_LSTM']:
            model_name = "LSTM"
        elif hyper_parameters['use_CNN_LSTM']:
            model_name = "CNN_LSTM"
        elif hyper_parameters['use_CNN']:
            model_name = "CNN"
        
        self.save_time = []
        self.results_cache = {}
        
        self.raw_df = self.make_df(hyper_parameters['data_start'], hyper_parameters['data_end'])
        print(f"\n원본 DF 로딩 완료: {self.raw_df.shape}\n")
        
        global_best_of_all_runs = {"rmse": float("inf"), "model": None, "features": None, "params_used": None}

        # --- ⭐️ [수정] 1. 그리드 서치 루프 (patience 루프 제거) ---
        for batch_size in hyper_parameters['batch_size']:
            for seq_length in hyper_parameters['seq_length']:
                for hidden_size in hyper_parameters['hidden_size']:
                    for num_layers in hyper_parameters['num_layers']:
                        
                        # ⭐️ 'patience' 루프 제거
                            
                        current_params = deepcopy(hyper_parameters)
                        current_params['batch_size'] = batch_size
                        current_params['seq_length'] = seq_length
                        current_params['hidden_size'] = hidden_size
                        current_params['num_layers'] = num_layers
                        # ⭐️ 'patience: 10'은 deepcopy로 자동 복사됨
                        
                        if hyper_parameters['use_LSTM']:
                            print(f"\n\n{'='*80}")
                            # ⭐️ [수정] current_params['patience'] 사용
                            print(f"🔥🔥🔥 [GRID SEARCH] batch: {batch_size}, seq: {seq_length}, hidden: {hidden_size}, layers: {num_layers}, patience: {current_params['patience']} 🔥🔥🔥")
                            print(f"{'='*80}")
                            
                            self.stepwise_log = []
                            grid_run_times = []
                            # ⭐️ [수정] current_params['patience'] 사용
                            log_header = f"GRID: batch:{batch_size}, seq:{seq_length}, hidden:{hidden_size}, layers:{num_layers}, patience:{current_params['patience']}\n"
                            self.stepwise_log.append(log_header)
                            
                            # ( ... 나머지 로직 동일 ... )
                            full_save_time_backup = deepcopy(self.save_time)
                            self.save_time = []
                            best_run_for_this_grid = self.run_stepwise_selection(current_params, model_name, set_seeds_func)
                            grid_run_times = deepcopy(self.save_time)
                            self.save_time = full_save_time_backup + grid_run_times
                            if best_run_for_this_grid["rmse"] < global_best_of_all_runs["rmse"]:
                                global_best_of_all_runs = deepcopy(best_run_for_this_grid)

                            print(f"\n... [GRID batch:{batch_size}, seq:{seq_length}...] 결과 저장 중 ...")

                            self.plot_train_val_loss(best_run_for_this_grid, model_name)
                            self.plot_predictions(best_run_for_this_grid, model_name)
                            self.save_txt(best_run_for_this_grid["params_used"], self.stepwise_log, model_name, best_run_for_this_grid, grid_run_times)

                            del best_run_for_this_grid

                            gc.collect()

                            if self.device.type == 'cuda':
                                torch.cuda.empty_cache()

                            
                        elif hyper_parameters['use_CNN_LSTM'] or hyper_parameters['use_CNN']:
                            for cnn_num_layers in hyper_parameters['cnn_num_layers']:
                                for num_filters in hyper_parameters['num_filters']:
                                    for kernel_size in hyper_parameters['kernel_size']:
                                        
                                        current_params['cnn_num_layers'] = cnn_num_layers
                                        current_params['num_filters'] = num_filters
                                        current_params['kernel_size'] = kernel_size
    
                                        print(f"\n\n{'='*80}")
                                        # ⭐️ [수정] current_params['patience'] 사용
                                        print(f"🔥🔥🔥 [GRID SEARCH] batch: {batch_size}, seq: {seq_length}, hidden: {hidden_size}, lstm_layers: {num_layers}, patience: {current_params['patience']}")
                                        print(f"                  cnn_layers: {cnn_num_layers}, filters: {num_filters}, kernel: {kernel_size} 🔥🔥🔥")
                                        print(f"{'='*80}")
    
                                        self.stepwise_log = []
                                        grid_run_times = []
                                        # ⭐️ [수정] current_params['patience'] 사용
                                        log_header = f"GRID: batch:{batch_size}, seq:{seq_length}, hidden:{hidden_size}, lstm_layers:{num_layers}, patience:{current_params['patience']}, cnn_layers:{cnn_num_layers}, filters:{num_filters}, kernel:{kernel_size}\n"
                                        self.stepwise_log.append(log_header)
                                        
                                        # ( ... 나머지 로직 동일 ... )
                                        full_save_time_backup = deepcopy(self.save_time)
                                        self.save_time = []
                                        best_run_for_this_grid = self.run_stepwise_selection(current_params, model_name, set_seeds_func)
                                        grid_run_times = deepcopy(self.save_time)
                                        self.save_time = full_save_time_backup + grid_run_times
                                        if best_run_for_this_grid["rmse"] < global_best_of_all_runs["rmse"]:
                                            global_best_of_all_runs = deepcopy(best_run_for_this_grid)
                                        print(f"\n... [GRID batch:{batch_size}, seq:{seq_length}...] 결과 저장 중 ...")
                                        self.plot_train_val_loss(best_run_for_this_grid, model_name)
                                        self.plot_predictions(best_run_for_this_grid, model_name)
                                        self.save_txt(best_run_for_this_grid["params_used"], self.stepwise_log, model_name, best_run_for_this_grid, grid_run_times)
                                        del best_run_for_this_grid
                                        gc.collect()
                                        if self.device.type == 'cuda':
                                            torch.cuda.empty_cache()

        # --- 3. 최종 요약 출력 ---
        # ( ... 동일 ... )
        print("\n\n==============================")
        print("🔥🔥🔥 모든 탐색 종료 🔥🔥🔥")
        print(f"🎉 'Global' 1등 Feature Set =", global_best_of_all_runs.get("features"))
        print(f"🎉 'Global' 1등 RMSE =", global_best_of_all_runs.get("rmse"))
        print(f"🎉 'Global' 1등 Params =", global_best_of_all_runs.get("params_used"))

        total_time_sec = sum(self.save_time)
        total_min, total_sec = divmod(total_time_sec, 60)
        print(f"🕒 총 학습 시간: {int(total_min)}분 {total_sec:.2f}초 ({total_time_sec:.2f} sec)")
        print(f"(캐시 제외 실제 학습 횟수: {len([t for t in self.save_time if t > 0])}회)")
        print("==============================")
        print("\n모든 작업이 완료되었습니다.")
    
    
    # ======================================================
    # 시각화 / 저장 함수 (⭐️ save_txt 수정)
    # ======================================================

    def plot_train_val_loss(self, best_overall, model_name):
        """ ⭐️ [수정] best_overall 딕셔너리를 통째로 받음 """
        
        train_losses = best_overall["train_losses"]
        val_losses = best_overall["val_losses"]
        rmse = best_overall["rmse"]
            
        plt.figure(figsize=(10,5))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.title('Train vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"./Output/{model_name}/(RMSE {rmse}) {model_name}_Result__Comparison with Loss.png")
        # plt.show()
        plt.close()


    # def plot_predictions(self, best_overall, model_name):
    #     """ ⭐️ [수정] best_overall 딕셔너리를 통째로 받음 """
        
    #     df = best_overall["df"]
    #     preds_inv = best_overall["preds_inv"]
    #     actual_inv = best_overall["actual_inv"]
    #     rmse = best_overall["rmse"]
    #     params = best_overall.get("params_used", {})
    #     seq_length = params.get('seq_length', 5) # params에서 seq_length 추출
    #     train_size = best_overall["train_size"]
    #     val_size = best_overall["val_size"]

    #     test_start = seq_length + train_size + val_size
    #     test_end = test_start + len(preds_inv)

    #     # ⭐️ 날짜 인덱싱 오류 방지
    #     if test_start >= len(df.index):
    #          print(f"[Warning] plot_predictions: test_start index ({test_start}) out of bounds.")
    #          test_start = len(df.index) - len(preds_inv)
    #          test_end = test_start + len(preds_inv)
    #          if test_start < 0:
    #              print("[Error] plot_predictions: Not enough data to plot.")
    #              return

    #     dates = df.index[test_start:test_end]
    #     plt.figure(figsize=(12,6))
    #     plt.plot(dates, actual_inv.ravel(), label='Actual (Test)', linewidth=1.5)
    #     plt.plot(dates, preds_inv.ravel(), label='Predicted (Test)', linewidth=1.5, alpha=0.8)
    #     plt.title(f'KOSPI Prediction using {model_name} (RMSE: {rmse})')
    #     plt.xlabel('Date')
    #     plt.ylabel('KOSPI Index')
    #     plt.legend()
    #     plt.grid(alpha=0.3)
    #     plt.gcf().autofmt_xdate()
    #     plt.savefig(f"./Output/{model_name}/(RMSE {rmse}) {model_name}_Result__KOSPI Prediction.png")
    #     # plt.show()
    #     plt.close()
    

    def plot_predictions(self, best_overall, model_name):
        """ ⭐️ [수정] best_overall 딕셔너리를 통째로 받음 """
        
        df = best_overall["df"]
        preds_inv = best_overall["preds_inv"]
        actual_inv = best_overall["actual_inv"]
        rmse = best_overall["rmse"]
        params = best_overall.get("params_used", {})
        seq_length = params.get('seq_length', 5) # params에서 seq_length 추출
        train_size = best_overall["train_size"]
        val_size = best_overall["val_size"]

        test_start = seq_length + train_size + val_size
        test_end = test_start + len(preds_inv)

        # ⭐️ 날짜 인덱싱 오류 방지 (기존 로직 유지)
        if test_start >= len(df.index):
            print(f"[Warning] plot_predictions: test_start index ({test_start}) out of bounds.")
            test_start = len(df.index) - len(preds_inv)
            test_end = test_start + len(preds_inv)
            if test_start < 0:
                print("[Error] plot_predictions: Not enough data to plot.")
                return

        dates = df.index[test_start:test_end]
        
        # ⭐️ 1. 시각화 데이터 추출 (가장 마지막 날짜와 값)
        actual_last = actual_inv[-1][0] 
        preds_last = preds_inv[-1][0]   
        
        plt.figure(figsize=(12,6))
        
        # ⭐️ [수정] 모든 데이터 포인트에 마커를 추가하여 개별 포인트를 식별 가능하게 함
        plt.plot(dates, actual_inv.ravel(), label='Actual (Test)', linewidth=1.5, color='blue', marker='o', markersize=3) 
        plt.plot(dates, preds_inv.ravel(), label='Predicted (Test)', linewidth=1.5, alpha=0.8, color='red', marker='x', markersize=3) 
        
        for date, actual, predicted in zip(dates, actual_inv.ravel(), preds_inv.ravel()):
            # 실제값 레이블
            plt.text(date, actual, f'{actual:.2f}', color='blue', fontsize=6, ha='right', va='center')
            # 예측값 레이블
            plt.text(date, predicted, f'{predicted:.2f}', color='red', fontsize=6, ha='left', va='center')
        
        # ⭐️ 2. 텍스트 주석 추가 (가장 최근 지점의 값만 표시 - 가독성 유지)
        # 실제값 (Actual) 최종 지점 표시
        plt.text(
            dates[-1], 
            actual_last, 
            f'{actual_last:.2f}', 
            color='blue', 
            fontsize=10, 
            ha='left', 
            va='bottom'
        )
        
        # 예측값 (Predicted) 최종 지점 표시
        plt.text(
            dates[-1], 
            preds_last, 
            f'{preds_last:.2f}', 
            color='red', 
            fontsize=10, 
            ha='left', 
            va='top' if preds_last > actual_last else 'bottom' # 겹치지 않도록 위치 조정
        )
        
        # ⭐️ 3. 타이틀 수정 (RMSE 포맷팅)
        plt.title(f'KOSPI Prediction using {model_name} (RMSE: {rmse:.4f})')
        plt.xlabel('Date')
        plt.ylabel('KOSPI Index')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.gcf().autofmt_xdate()
        # plt.tight_layout() 
        plt.savefig(f"./Output/{model_name}/(RMSE {rmse}) {model_name}_Result__KOSPI Prediction.png")
        # plt.show()
        plt.close()
        
    
    def save_txt(self, current_params_log, stepwise_log, model_name, best_overall, grid_save_time):
        """
        ⭐️ [수정]
        - hyper_parameters -> current_params_log (단일 값 딕셔너리)
        - save_time -> grid_save_time (이 그리드의 시간 리스트)
        """
        
        total_seconds = sum(grid_save_time) # ⭐️ 이 그리드의 시간
        minutes, seconds = divmod(total_seconds, 60)



        # # ⭐️ 새로 추가된 예측 CSV 저장 로직
        # df = best_overall["df"]
        # preds_inv = best_overall["preds_inv"]
        # actual_inv = best_overall["actual_inv"]
        # params = best_overall.get("params_used", {})
        # seq_length = params.get('seq_length', 5) 
        # train_size = best_overall["train_size"]
        # val_size = best_overall["val_size"]
        # rmse = best_overall["rmse"]
        
        # test_start = seq_length + train_size + val_size
        # test_end = test_start + len(preds_inv)
        
        # # # 날짜 인덱싱 안전 처리 (plot_predictions와 동일)
        # # if test_start >= len(df.index):
        # #      test_start = len(df.index) - len(preds_inv)
        # #      test_end = test_start + len(preds_inv)
        # #      if test_start < 0:
        # #          # CSV 저장이 불가능한 경우, 텍스트 파일만 저장합니다.
        # #          csv_save_path = "N/A (Not enough data to plot/save)"
        # #      else:
        # #          dates = df.index[test_start:test_end]
        # #          df_results = pd.DataFrame({'Actual': actual_inv.ravel(), 'Predicted': preds_inv.ravel()}, index=dates)
        # #          csv_save_path = f"./Output/{model_name}/(RMSE {rmse}) {model_name}_Result__Predictions.csv"
        # #          df_results.to_csv(csv_save_path)
        # #          print(f"✅ Prediction results saved to {csv_save_path}")
        # # else:
        # dates = df.index[test_start:test_end]
        # df_results = pd.DataFrame({'Actual': actual_inv.ravel(), 'Predicted': preds_inv.ravel()}, index=dates)
        # csv_save_path = f"./Output/{model_name}/(RMSE {rmse}) {model_name}_Result__Predictions.csv"
        # df_results.to_csv(csv_save_path, encoding='EUC-KR')
        # print(f"✅ Prediction results saved to {csv_save_path}")



        save_path = f"./Output/{model_name}/(RMSE {best_overall['rmse']}) {model_name}_Result__HyperParameters.txt"
        
        config_text = """"""
        
        # if current_params_log['step_wise']:
        #     config_text += f"""
        #     # ============================
        #     # Stepwise Feature Selection Log
        #     # ============================
        #     (Base/Candidates는 Experiment Configuration 섹션 참조)
        #     """
            
        config_text += f"""
        # ============================
        # Experiment Result (For This Grid)
        # ============================
        
        👍 This Grid Process Time:
        -> {total_seconds} sec
        -> {int(minutes)}분 {seconds:.2f}초
        
        📋 Final Data Columns:
        {best_overall["features"]}

        ✅ Final RMSE: {best_overall["rmse"]}


        # ============================
        # Experiment Configuration
        # ============================
        
        🧩 Model Used: **{model_name}**
        
        📅 Data Range
        - Start Date: {current_params_log['data_start']}
        - End Date  : {current_params_log['data_end']}
        - test_start_date: {current_params_log['test_start_date']}
        - Sequence Length: {current_params_log['seq_length']}

        📊 Data Split
        - Train Ratio: {current_params_log['train_ratio']}
        - Validation Ratio: {current_params_log['val_ratio']}

        🧠 Model Parameters
        - Hidden Size: {current_params_log['hidden_size']}
        - Num Layers : {current_params_log['num_layers']}
        - Num Classes: {current_params_log['num_classes']}
        """

        if current_params_log['use_CNN_LSTM']:
             config_text += f"""
        🧠 CNN Parameters
        - CNN Num Layers: {current_params_log['cnn_num_layers']}
        - Num Filters   : {current_params_log['num_filters']}
        - Kernel Size   : {current_params_log['kernel_size']}
        """
        
        config_text += f"""
        ⚙️ Training Setup
        - Batch Size   : {current_params_log['batch_size']}
        - Epochs       : {current_params_log['epochs']}
        - Learning Rate: {current_params_log['learning_rate']}
        
        # ============================
        # Base/Candidate Features
        # ============================
        - BASE_FEATURES: {current_params_log['BASE_FEATURES']}
        - CANDIDATES   : {current_params_log['CANDIDATES']}
        """

        # if current_params_log['step_wise']:
        #     config_text += """
            
        # # ============================
        # # Log Data (For This Grid)
        # # ============================
        # """
        #     for l in stepwise_log:
        #         config_text += "  " + l + "\n"
                
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(config_text)