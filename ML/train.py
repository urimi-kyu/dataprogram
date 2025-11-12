import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler 
import joblib 

# --- 0. 모델 아키텍처 정의 ---
# (변경 없음)
class MDDModel(nn.Module):
    def __init__(self, input_dim=8, output_dim=12):
        super(MDDModel, self).__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )
    def forward(self, x):
        return self.network(x)

# --- 1. CSV 파일 로드 및 전처리 ---
def load_and_preprocess_data(file_path, disaster_col, label_cols, disaster_order_list):
    try:
        df = pd.read_csv(file_path, encoding='cp949')
    except FileNotFoundError:
        print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 오류: CSV 로드 실패 (인코딩 등): {e}")
        sys.exit(1)
    
    # [데이터 전처리 1] "자연재해(홍수)" -> "홍수"
    try:
        # (EventType이 비어있는 행(NaN)이 있을 수 있으므로 .dropna() 추가)
        df = df.dropna(subset=[disaster_col]) 
        df[disaster_col] = df[disaster_col].apply(lambda x: str(x).split('(')[-1].replace(')', ''))
    except Exception as e:
        print(f"❌ 오류: 'EventType' 컬럼 정리 중 오류 발생: {e}")
        sys.exit(1)

    # [❗ 수정됨] 데이터 전처리 2: MDD 컬럼의 빈 칸(NaN)을 0으로 채우기
    try:
        df[label_cols] = df[label_cols].fillna(0)
        # (혹시 숫자가 아닌 값이 섞여있을 경우를 대비해 강제 변환)
        df[label_cols] = df[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
    except Exception as e:
        print(f"❌ 오류: MDD 컬럼(Y)의 NaN 값을 0으로 채우는 중 오류 발생: {e}")
        sys.exit(1)

    # 2. 라벨(Y) 데이터 추출 (12개 MDD 컬럼)
    try:
        Y_data_raw = df[label_cols].values.astype(np.float32)
        
        # (NaN이 0으로 채워졌는지 최종 확인)
        if np.isnan(Y_data_raw).any():
            print("❌ 오류: Y 데이터에 여전히 NaN이 남아있습니다. CSV 파일을 확인하세요.")
            sys.exit(1)
            
    except KeyError as e:
        print(f"❌ 오류: CSV에 MDD 컬럼({e})이 없습니다. LABEL_COLUMNS를 확인하세요.")
        sys.exit(1)
        
    # [Y 데이터 정규화]
    print("   (Y 데이터(MDD) 정규화 중...)")
    scaler = StandardScaler()
    Y_data_scaled = scaler.fit_transform(Y_data_raw)
    
    # (스케일링 후 NaN이 되었는지 확인 - 분산이 0인 컬럼이 있는지)
    if np.isnan(Y_data_scaled).any():
        print("❌ 오류: Y 데이터 스케일링 후 NaN이 발생했습니다.")
        print("   (CSV에서 모든 사건의 MDD가 동일한 컬럼이 있는지 확인하세요.)")
        sys.exit(1)
        
    scaler_save_path = "mdd_scaler.joblib"
    joblib.dump(scaler, scaler_save_path)
    print(f"   (Y 정규화 스케일러를 '{scaler_save_path}'에 저장했습니다.)")
        
    # 5. 피처(X) 데이터 전처리 (One-Hot Encoding)
    try:
        features_df = pd.get_dummies(df, columns=[disaster_col], prefix="", prefix_sep="")
        X_df = features_df.reindex(columns=disaster_order_list, fill_value=0)
        X_data = X_df.values.astype(np.float32)
    except Exception as e:
        print(f"❌ 오류: X 데이터 전처리 중 오류 발생 - {e}")
        sys.exit(1)

    # 6. PyTorch 텐서로 변환
    X_tensor = torch.tensor(X_data)
    Y_tensor = torch.tensor(Y_data_scaled) # (정규화된 Y 데이터 사용)
    
    print(f"✅ CSV 파일 로드 및 전처리 완료: '{file_path}'")
    print(f"   (총 {len(df)}개 사건, 피처 {X_tensor.shape[1]}개, 라벨 {Y_tensor.shape[1]}개)")
    
    return X_tensor, Y_tensor

# --- 2. 학습 설정 ---
INPUT_DIM = 8   
OUTPUT_DIM = 12 
NUM_EPOCHS = 100
BATCH_SIZE = 8 
MODEL_SAVE_PATH = "mdd_model.pth"
LEARNING_RATE = 0.0001 

CSV_FILE_PATH = "event_result.csv" 
DISASTER_TYPE_COLUMN = "EventType" 

DISASTER_ORDER_LIST = [
    "산불", "지진", "태풍", "감염병", 
    "가뭄", "폭설", "홍수", "기타"
]
LABEL_COLUMNS = [
    "Market (KOSPI)", "KOSPI 200 - Communication Services", "KOSPI 200 - Construction",
    "KOSPI 200 - Heavy Industry", "KOSPI 200 - Steel/Materials", "KOSPI 200 - Energy/Chemicals",
    "KOSPI 200 - Information Technology", "KOSPI 200 - Finance", "KOSPI 200 - Consumer Staples",
    "KOSPI 200 - Consumer Discretionary", "KOSPI 200 - Industrials", "KOSPI 200 - Healthcare"
] 

# --- 3. 학습 실행 ---
if __name__ == "__main__":
    
    if len(DISASTER_ORDER_LIST) != INPUT_DIM:
        print(f"❌ 설정 오류: DISASTER_ORDER_LIST는 {INPUT_DIM}개여야 합니다.")
        sys.exit(1)
    if len(LABEL_COLUMNS) != OUTPUT_DIM:
        print(f"❌ 설정 오류: LABEL_COLUMNS 리스트에 {OUTPUT_DIM}개가 필요합니다.")
        sys.exit(1)

    X_train, Y_train = load_and_preprocess_data(
        CSV_FILE_PATH, 
        DISASTER_TYPE_COLUMN, 
        LABEL_COLUMNS,
        DISASTER_ORDER_LIST
    )
    
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MDDModel(INPUT_DIM, OUTPUT_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- 모델 학습 시작 ---")
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                print(f"❌ 오류: Epoch {epoch+1}에서 Loss가 NaN이 되었습니다. 학습을 중단합니다.")
                sys.exit(1) # (NaN 감지기 유지)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")

    print("--- 모델 학습 완료 ---")
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"모델이 '{MODEL_SAVE_PATH}' 파일로 저장되었습니다.")