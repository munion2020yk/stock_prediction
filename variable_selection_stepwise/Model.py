import torch
import torch.nn as nn

# ======================================================
# 1. LSTM 모델 정의 (이전과 동일, 완성형)
# ======================================================

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        """
        num_layers 파라미터를 사용하여 지정된 수의 LSTM 층을 자동으로 쌓습니다.
        """
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # nn.LSTM이 num_layers를 인자로 받아 자동으로 여러 층을 쌓아줍니다.
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,      # (예: 1, 2, 3...)
            batch_first=True
        )
        
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # (hn, cn) shape: (num_layers, batch, hidden_size)
        out, (hn, cn) = self.lstm(x)
        
        # 마지막 층(layer)의 마지막 시점(time step) 은닉 상태만 사용
        hn_last_layer = hn[-1]  # (batch, hidden_size)
        
        # FC Layer 통과
        out = self.relu(hn_last_layer)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        
        return out


# ======================================================
# 2. CNN + LSTM 모델 정의 (⭐ cnn_num_layers 적용)
# ======================================================
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length,
                 cnn_num_layers=1, num_filters=64, kernel_size=3):
        """
        Args:
            num_layers (int): 쌓을 "LSTM" 층의 개수
            cnn_num_layers (int): 쌓을 "CNN" 층의 개수
        """
        super(CNN_LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers      # LSTM 층 수
        self.seq_length = seq_length
        self.cnn_num_layers = cnn_num_layers # CNN 층 수
        self.num_filters = num_filters

        # 🔹 1. CNN Stack (동적으로 생성)
        self.cnn_stack = nn.ModuleList()
        
        current_in_channels = input_size # 첫 번째 CNN 층의 입력 채널
        
        for i in range(cnn_num_layers):
            # Conv1d 추가
            self.cnn_stack.append(
                nn.Conv1d(
                    in_channels=current_in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # 시퀀스 길이 유지
                )
            )
            # BatchNorm1d 추가
            self.cnn_stack.append(nn.BatchNorm1d(num_filters))
            # ReLU 추가
            self.cnn_stack.append(nn.ReLU())
            
            # 다음 CNN 층의 입력 채널은
            # 현재 층의 출력 채널(num_filters)이 됩니다.
            current_in_channels = num_filters 

        # 🔹 2. LSTM Module
        # (중요) LSTM의 input_size는 CNN 스택의 최종 출력 채널 수(num_filters)
        self.lstm_module = LSTM(
            num_classes=num_classes,
            input_size=num_filters,   # ⭐️ CNN의 최종 출력을 입력으로 받음
            hidden_size=hidden_size,
            num_layers=num_layers,    # ⭐️ LSTM 층 수를 여기에 전달
            seq_length=seq_length
        )

    def forward(self, x):
        # -----------------------------
        # 입력 x: (batch, seq_len, input_size)
        # -----------------------------
        
        # 1. CNN 입력용으로 변환: (batch, input_size, seq_len)
        x_cnn = x.permute(0, 2, 1)
        
        # 2. 동적으로 생성된 CNN 스택 모두 통과
        for layer in self.cnn_stack:
            x_cnn = layer(x_cnn)
        # x_cnn shape: (batch, num_filters, seq_len)
        
        # 3. LSTM 입력용으로 변환: (batch, seq_len, num_filters)
        x_lstm_in = x_cnn.permute(0, 2, 1)
        
        # 4. LSTM 모듈 통과 (LSTM + FC Layers)
        out = self.lstm_module(x_lstm_in)
        
        return out
    
    

class CNNModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, cnn_num_layers=1, num_filters=64, kernel_size=3, output_size = 1):
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
