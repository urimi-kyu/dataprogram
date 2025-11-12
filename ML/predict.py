import torch
import torch.nn as nn
import numpy as np
import json
import sys
import joblib # 1. Scaler 로드를 위해 joblib 임포트

# --- 0. 모델 아키텍처 정의 ---
# (train.py와 동일해야 함)
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

# --- 1. 모델 및 스케일러 로드 ---
INPUT_DIM = 8
OUTPUT_DIM = 12 
MODEL_LOAD_PATH = "mdd_model.pth" 
SCALER_LOAD_PATH = "mdd_scaler.joblib" # 2. 스케일러 파일 경로

# 모델 로드
model = MDDModel(INPUT_DIM, OUTPUT_DIM)
try:
    model.load_state_dict(torch.load(MODEL_LOAD_PATH))
except FileNotFoundError:
    print(json.dumps({"error": f"Model file not found: {MODEL_LOAD_PATH}"}), file=sys.stderr)
    sys.exit(1)
except RuntimeError as e:
    print(json.dumps({"error": f"Model load error: {e}. 'train.py'를 다시 실행하세요."}), file=sys.stderr)
    sys.exit(1)
model.eval()

# [❗ 3. 중요] 스케일러 로드
try:
    scaler = joblib.load(SCALER_LOAD_PATH)
except FileNotFoundError:
    print(json.dumps({"error": f"Scaler file not found: {SCALER_LOAD_PATH}. 'train.py'를 실행하세요."}), file=sys.stderr)
    sys.exit(1)


# --- 3. 예측 수행 함수 ---
def predict_mdd(disaster_vector, loaded_model, loaded_scaler):
    with torch.no_grad():
        input_tensor = disaster_vector.unsqueeze(0) 
        
        # 4. 모델이 '정규화된(작은)' 값 예측
        scaled_prediction = loaded_model(input_tensor) # (예: [-1.5, 0.8, ...])
        
        # 5. [❗ 중요] 스케일러를 사용해 '원래(큰)' MDD 값으로 복원
        original_prediction = loaded_scaler.inverse_transform(scaled_prediction.numpy())
        
        # (1, 12) -> (12,)
        return original_prediction.squeeze(0)

# --- 4. 메인 실행 (서비스 로직) ---
if __name__ == "__main__":
    
    expected_args = INPUT_DIM
    if len(sys.argv) != expected_args + 1:
        error_msg = f"Error: Expected {expected_args} disaster values, but got {len(sys.argv) - 1}"
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)

    try:
        input_values = [float(arg) for arg in sys.argv[1:]]
        current_vector = torch.tensor(input_values).float()
    except ValueError:
        error_msg = "Error: All input arguments must be numeric values."
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)

    # 6. 예측 (스케일러와 모델 전달)
    predicted_mdds_original = predict_mdd(current_vector, model, scaler)
    
    # 7. Python list로 변환 (복원된 값이므로 음수 등 큰 값이 나와야 함)
    mdd_values_list = np.round(predicted_mdds_original, 2).tolist()
    
    # 8. JSON 형식으로 12개 값 리스트를 출력
    print(json.dumps(mdd_values_list))