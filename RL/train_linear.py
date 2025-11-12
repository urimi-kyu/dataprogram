import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
import joblib

# --- 1. CSV 파일 로드 및 전처리 ---
def load_and_preprocess_data(file_path, disaster_col, label_cols, disaster_order_list):
    """
    event_result.csv를 로드하고 One-Hot-Encoding하여 numpy array로 반환합니다.
    """
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
        df = df.dropna(subset=[disaster_col]) 
        df[disaster_col] = df[disaster_col].apply(lambda x: str(x).split('(')[-1].replace(')', ''))
    except Exception as e:
        print(f"❌ 오류: 'EventType' 컬럼 정리 중 오류 발생: {e}")
        sys.exit(1)

    # [데이터 전처리 2] MDD 컬럼의 빈 칸(NaN)을 0으로 채우기
    try:
        df[label_cols] = df[label_cols].fillna(0)
        df[label_cols] = df[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    except Exception as e:
        print(f"❌ 오류: MDD 컬럼(Y)의 NaN 값을 0으로 채우는 중 오류 발생: {e}")
        sys.exit(1)

    # 2. 라벨(Y) 데이터 추출 (12개 MDD 컬럼)
    try:
        Y_data = df[label_cols].values.astype(np.float32)
    except KeyError as e:
        print(f"❌ 오류: CSV에 MDD 컬럼({e})이 없습니다. LABEL_COLUMNS를 확인하세요.")
        sys.exit(1)
        
    # 4. 피처(X) 데이터 전처리 (One-Hot Encoding)
    try:
        features_df = pd.get_dummies(df, columns=[disaster_col], prefix="", prefix_sep="")
        X_df = features_df.reindex(columns=disaster_order_list, fill_value=0)
        X_data = X_df.values.astype(np.float32)
    except Exception as e:
        print(f"❌ 오류: X 데이터 전처리 중 오류 발생 - {e}")
        sys.exit(1)

    # 5. 최종 데이터 확인
    if np.isnan(Y_data).any():
        print("❌ 오류: Y 데이터에 NaN이 남아있습니다. CSV 파일을 확인하세요.")
        sys.exit(1)

    print(f"✅ CSV 파일 로드 및 전처리 완료: '{file_path}'")
    print(f"   (총 {len(df)}개 사건, 피처 {X_data.shape[1]}개, 라벨 {Y_data.shape[1]}개)")
    
    return X_data, Y_data

# --- 2. 학습 설정 ---
INPUT_DIM = 8   
OUTPUT_DIM = 12 

# [❗ 수정됨] 모델 저장 파일명 변경
MODEL_SAVE_PATH = "mdd_linear_model.joblib" 

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
    
    model = LinearRegression()
    
    print("\n--- 선형 회귀 모델 학습 시작 ---")
    
    model.fit(X_train, Y_train)

    print("--- 모델 학습 완료 ---")
    
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"모델이 '{MODEL_SAVE_PATH}' 파일로 저장되었습니다.")