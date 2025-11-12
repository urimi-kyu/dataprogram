import torch
import torch.nn as nn
import numpy as np
import os
import joblib
import sys

# ------------------------------------------------------------------
# --- 0. 모델 아키텍처 정의 (mdd_model.pth와 동일해야 함) ---
# ------------------------------------------------------------------
# ⚠️ mdd_prediction_service.py와 동일한 상수를 사용해야 함
INPUT_DIM = 8
OUTPUT_DIM = 12


class MDDModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
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


# ------------------------------------------------------------------
# --- 1. 모델 및 스케일러 로드 (캐싱) ---
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LOAD_PATH = os.path.join(BASE_DIR, "mdd_model.pth")
SCALER_LOAD_PATH = os.path.join(BASE_DIR, "mdd_scaler.joblib")

LOADED_MODEL = None
LOADED_SCALER = None


def load_mdd_assets():
    """모델과 스케일러를 모두 로드하고 캐시합니다."""
    global LOADED_MODEL, LOADED_SCALER
    if LOADED_MODEL is None or LOADED_SCALER is None:
        try:
            # 1. 모델 로드
            model = MDDModel(INPUT_DIM, OUTPUT_DIM)
            model.load_state_dict(torch.load(
                MODEL_LOAD_PATH, map_location=torch.device('cpu')))
            model.eval()
            LOADED_MODEL = model

            # 2. 스케일러 로드
            LOADED_SCALER = joblib.load(SCALER_LOAD_PATH)

        except FileNotFoundError as e:
            # 필수 파일이 없는 경우, 예측 기능을 비활성화하기 위해 오류 발생
            raise FileNotFoundError(
                f"모델/스케일러 파일 로드 실패: {e.filename} 파일을 확인하세요.")
        except Exception as e:
            raise Exception(f"MDD 모델 에셋 로드 중 오류 발생: {e}")

    return LOADED_MODEL, LOADED_SCALER

# ------------------------------------------------------------------
# --- 2. 예측 수행 함수 (외부 호출용) ---
# ------------------------------------------------------------------


def predict_mdd_value(feature_vector: np.ndarray):
    """
    MDD를 예측합니다. (스케일링 및 역변환 포함)
    """
    model, scaler = load_mdd_assets()

    # 1. 스케일링 (입력 피처)
    scaled_input_features = scaler.transform(feature_vector)

    # 2. 텐서로 변환
    input_tensor = torch.from_numpy(scaled_input_features).float()

    with torch.no_grad():
        # 3. 모델 예측 (scaled prediction)
        scaled_prediction = model(input_tensor)

        # 4. 역변환 (원래 MDD 값으로 복원)
        scaled_prediction_np = scaled_prediction.cpu().numpy()
        original_prediction = scaler.inverse_transform(scaled_prediction_np)

        return original_prediction.squeeze(0)


# 외부에서 사용할 상수 노출
INPUT_DIM = INPUT_DIM
