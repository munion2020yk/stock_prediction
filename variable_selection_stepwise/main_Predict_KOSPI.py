# main_Predict_KOSPI.py

import torch
import time
import pandas as pd
# (Model 임포트 불필요)
from Function_Collection import function_collection
from copy import deepcopy
import os


# ======================================================
# 0. 디바이스 / 랜덤시드
# ======================================================
def set_seeds(seed_value=100):
    """결과 재현을 위해 랜덤 시드를 고정하는 함수"""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
set_seeds(100)


# ======================================================
# 1. 공통 파라미터 설정 (그리드 서치용 리스트)
# ======================================================

hyper_parameters = {
    
    "step_wise": True,

    "use_LSTM": False,
    "use_CNN_LSTM": False,
    "use_CNN": True,

    "BASE_FEATURES" : ['Open', 'High', 'Low', 'Close', 'Volume'],
    
    # "CANDIDATES" : ['Change', 'UpDown', 'Comp', 'Amount', 'MarCap', 'USD_KRW', 'EUR_KRW', 'NASDAQ', 'VKOSPI_Close', 'VKOSPI_Open', 'VKOSPI_High', 'VKOSPI_Low', 'VKOSPI_Change', 'KOSPI_Future_Close', 'KOSPI_Future_Open', 'KOSPI_Future_High', 'KOSPI_Future_Low', 'KOSPI_Future_Change', 'WTI_Close', 'WTI_Open', 'WTI_High', 'WTI_Low', 'WTI_Change', 'Foreign_Holdings_ratio'],
    
    # "CANDIDATES" : ['Change', 'UpDown', 'Comp', 'Amount', 'MarCap'],

    # "CANDIDATES" : ['Change'],
    
    "CANDIDATES" : ['KOSPI_Future_Change', 'NASDAQ', 'USD_KRW', 'VKOSPI_Change', 'KOSPI_Future_Low'],
    
    "data_start": "2013-08-06",     # 시작 날짜
    "data_end": "2025-11-28",       # 종료 날짜

    "test_start_date": "2025-11-24",

    "train_ratio": 0.7,             # 학습 데이터 비율
    "val_ratio": 0.3,               # 검증 데이터 비율
    
    # 공통 파라미터
    "seq_length": [5, 7, 10],            # 시퀀스 길이(날짜)
    "hidden_size": [128, 256],         # 은닉층 크기 256
    "num_layers": [1, 2],             # LSTM 레이어 수
    "num_classes": 1,                 # 출력 레이어 크기
    
    # CNN 파라미터
    "cnn_num_layers":[1, 2],          # CNN 레이어 수
    "num_filters":[32, 64],           # 필터의 수 
    "kernel_size":[3, 5, 10],             # 커널의 크기 

    "batch_size": [256],           # 배치 사이즈
    "epochs": 100,                    # 에폭
    "patience":10,                    # early stopping 기준 
    "learning_rate": 0.005            # learning rate 
    
    
    # # 공통 파라미터
    # "seq_length": [5],            # 시퀀스 길이(날짜)
    # "hidden_size": [256],         # 은닉층 크기 256
    # "num_layers": [1],             # LSTM 레이어 수
    # "num_classes": 1,                 # 출력 레이어 크기

    # "batch_size": [256],           # 배치 사이즈
    # "epochs": 100,                    # 에폭
    # "patience":10,                    # early stopping 기준 
    # "learning_rate": 0.005            # learning rate 
}


# ======================================================
# 2. 메인 실행
# ======================================================
if __name__ == "__main__":
    
    func = function_collection()
    
    # ⭐️ Function_Collection의 메인 컨트롤 타워 함수 호출
    func.run_grid_search_experiment(
        hyper_parameters=hyper_parameters,
        set_seeds_func=set_seeds 
    )
    
    
    # columns = hyper_parameters['BASE_FEATURES'] + hyper_parameters['CANDIDATES']
    # result = func._train_and_eval(columns, hyper_parameters, set_seeds)
    # print(result)
    