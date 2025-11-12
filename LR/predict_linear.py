import numpy as np
import json
import sys
import joblib 

# --- 1. 모델 로드 ---
# [❗ 수정됨] 모델 로드 파일명 변경
MODEL_LOAD_PATH = "mdd_linear_model.joblib" 

try:
    model = joblib.load(MODEL_LOAD_PATH)
except FileNotFoundError:
    print(json.dumps({"error": f"Model file not found: {MODEL_LOAD_PATH}. 'train_linear.py'를 실행하세요."}), file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": f"Model load error: {e}"}), file=sys.stderr)
    sys.exit(1)


# --- 3. 예측 수행 함수 ---
def predict_mdd(disaster_vector, loaded_model):
    
    try:
        # 입력 벡터(파이썬 리스트)를 (1, 8) 형태의 NumPy Array로 변환
        input_array = np.array(disaster_vector).reshape(1, -1)
    except Exception as e:
        print(f"Input vector error: {e}", file=sys.stderr)
        return None

    # 선형 회귀 모델로 예측 (결과는 (1, 12) 형태)
    prediction = loaded_model.predict(input_array)
    
    # (1, 12) -> (12,)
    return prediction.squeeze(0)

# --- 4. 메인 실행 (서비스 로직) ---
if __name__ == "__main__":
    
    INPUT_DIM = 8 # (입력 차원은 8)
    
    expected_args = INPUT_DIM
    if len(sys.argv) != expected_args + 1:
        error_msg = f"Error: Expected {expected_args} disaster values, but got {len(sys.argv) - 1}"
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)

    try:
        # [0.1, 0.8, ...] 형태의 파이썬 리스트
        input_values = [float(arg) for arg in sys.argv[1:]]
    except ValueError:
        error_msg = "Error: All input arguments must be numeric values."
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)

    # 예측 (모델 전달)
    predicted_mdd_numpy = predict_mdd(input_values, model)
    
    if predicted_mdd_numpy is not None:
        # Python list로 변환
        mdd_values_list = np.round(predicted_mdd_numpy, 2).tolist()
        
        # JSON 형식으로 12개 값 리스트를 출력
        print(json.dumps(mdd_values_list))
    else:
        print(json.dumps({"error": "Prediction failed."}), file=sys.stderr)
        sys.exit(1)