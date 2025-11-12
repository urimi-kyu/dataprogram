import urllib.request
import json
import datetime
import subprocess # 1. AI 모델(predict.py) 호출을 위해 임포트
import sys
import os # 2. 파일 경로 관리를 위해 임포트

# --- 1. 기존 뉴스 크롤링 코드 (Notebook의 코드) ---

# (중요!) Naver API 인증 정보 (환경 변수나 설정 파일에서 관리하는 것을 권장)
CLIENT_ID = "xAkr8e4sWLnQW4x_YnIh"
CLIENT_SECRET = "gkDg6FlbBR"

TODAY_DATE_STR = datetime.date.today().strftime("%Y%m%d")

def search_disaster_occurrence_news(keyword):
    """
    네이버 뉴스 API를 호출하여 특정 키워드의 오늘 뉴스 건수를 반환합니다.
    """
    encText = urllib.parse.quote(f"'{keyword}' AND (속보 OR 발생 OR 피해)")

    # 오늘 날짜로 검색 범위 제한 (startdate=...&enddate=...)
    url = (f"https://openapi.naver.com/v1/search/news.json?query={encText}"
           f"&display=10&sort=sim&startdate={TODAY_DATE_STR}&enddate={TODAY_DATE_STR}")

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", CLIENT_SECRET)

    total_count = 0
    try:
        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if rescode == 200:
            response_body = response.read()
            result = json.loads(response_body.decode('utf-8'))
            total_count = int(result.get('total', 0))
            print(f"✅ [{keyword}] 키워드 검색 완료. (오늘 뉴스: {total_count}건)")

        else:
            print(f"❌ [{keyword}] API Error Code: {rescode}")

    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error: {e.code} ({e.reason}). 인증 정보(Client ID/Secret)를 다시 확인하세요.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

    return total_count

# --- 2. [추가된 코드] AI 모델 연동을 위한 로직 ---

def convert_counts_to_vector(disaster_counts):
    """
    '뉴스 건수' 딕셔너리를 '재난 강도 벡터(0.0~1.0)' 리스트로 변환합니다.
    (이 함수는 모델 성능에 매우 중요하므로 고도화가 필요합니다!)

    모델 입력 순서: [산불, 지진, 태풍, 감염병, 가뭄, 폭설, 홍수, 기타]
    """

    # 1. 모델의 입력 순서대로 키를 정렬
    # (주의: Notebook의 '폭우'를 '홍수'로 매핑)
    model_input_order = [
        "산불", "지진", "태풍", "감염병", "가뭄", "폭설", "홍수", "기타"
    ]

    # '폭우' 건수는 '홍수'에 합산 (예시)
    flood_count = disaster_counts.get("홍수", 0) + disaster_counts.get("폭우", 0)
    disaster_counts["홍수"] = flood_count

    vector = []

    # 2. 정규화(Normalization) 로직
    # (예시: 100건 이상이면 최대 강도(1.0)로 간주, 그 이하는 비례)
    # (이 로직을 정교화해야 모델 예측이 정확해집니다)
    MAX_NEWS_COUNT_FOR_SCORE_1 = 100.0

    for disaster_name in model_input_order:
        count = disaster_counts.get(disaster_name, 0) # 해당 재난 키가 없으면 0

        # 0 ~ 1.0 사이의 값으로 정규화
        score = min(count / MAX_NEWS_COUNT_FOR_SCORE_1, 1.0)

        vector.append(score)

    return vector

def get_mdd_prediction(disaster_vector_list):
    """
    AI 모델(predict.py)을 subprocess로 실행하고, 12개 MDD 예측값을 받아옵니다.

    :param disaster_vector_list: [0.1, 0.0, ..., 0.2] 형태의 8개 점수 리스트
    """
    print("\n--- AI 모델 호출 시작 ---")

    # 3. [경로 설정] AI 모델의 위치
    python_executable = r"c:\Python313\python.exe"
    
    # ❗ [수정됨] predict.py 스크립트의 경로를 'Downloads' 폴더로 수정
    script_path = r"C:\Users\hotba\Downloads\predict.py"

    # 4. 인자 리스트 생성 (숫자를 문자열로 변환)
    args = [str(round(val, 3)) for val in disaster_vector_list]

    # 5. 명령어 조합
    command = [python_executable, script_path] + args

    print(f"실행 명령어: {' '.join(command)}")

    try:
        # 6. 스크립트 실행
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )

        # 7. 성공: stdout(표준 출력)을 JSON으로 파싱
        # predict.py가 출력한 "[5.21, ..., 6.05]"
        mdd_values = json.loads(result.stdout)
        print("✅ AI 모델 예측 성공:")
        return mdd_values

    except subprocess.CalledProcessError as e:
        # 8. 실패: predict.py가 오류(stderr)를 출력한 경우
        error_info_json = e.stderr
        print(f"❌ AI 모델 스크립트 실행 오류: {error_info_json}")
        return None
    except Exception as e:
        # 9. 기타 오류 (경로 문제, JSON 파싱 실패 등)
        print(f"❌ 백엔드 실행 오류: {e}")
        return None

# --- 3. [메인 로직] 전체 워크플로우 실행 ---
if __name__ == "__main__":

    print(f"--- {TODAY_DATE_STR} 재난 상황 분석 시작 ---")

    # 1. 크롤링할 재난 키워드 정의
    # (모델이 사용하는 8종류 + Notebook에 있던 '폭우' 포함)
    disasters_to_check = [
        "산불", "지진", "태풍", "감염병", "가뭄", "폭설", "홍수", "기타",
        "폭우" # '홍수'로 합산하기 위해 검색
    ]

    disaster_counts = {}

    for disaster in disasters_to_check:
        count = search_disaster_occurrence_news(disaster)
        disaster_counts[disaster] = count

    print("\n--- 재난 벡터 변환 중 ---")

    # 2. 크롤링 결과(건수)를 AI 모델 입력 벡터(점수)로 변환
    disaster_vector = convert_counts_to_vector(disaster_counts)

    print(f"생성된 재난 강도 벡터 (8차원): {disaster_vector}")

    # 3. AI 모델 호출하여 MDD 예측
    mdd_predictions = get_mdd_prediction(disaster_vector)

    # 4. 최종 결과 처리
    if mdd_predictions:
        print("\n========================================")
        print("     📈 최종 MDD 예측 결과 (12 섹터) 📈")
        print("========================================")

        # 12개 섹터 이름 (predict.py와 순서 동일)
        # (train.py의 LABEL_COLUMNS 순서와 일치해야 함)
        sectors = [
            "Market (KOSPI)",                     # (1001)
            "KOSPI 200 - Communication Services", # (1150)
            "KOSPI 200 - Construction",           # (1151)
            "KOSPI 200 - Heavy Industry",         # (1152)
            "KOSPI 200 - Steel/Materials",        # (1153)
            "KOSPI 200 - Energy/Chemicals",       # (1154)
            "KOSPI 200 - Information Technology", # (1155)
            "KOSPI 200 - Finance",                # (1156)
            "KOSPI 200 - Consumer Staples",       # (1157)
            "KOSPI 200 - Consumer Discretionary", # (1158)
            "KOSPI 200 - Industrials",            # (1159)
            "KOSPI 200 - Healthcare"              # (1160)
        ]

        # (만약 mdd_predictions 개수와 sectors 개수가 다르면 오류가 날 수 있음)
        if len(mdd_predictions) != len(sectors):
            print(f"❌ 오류: 예측된 MDD 개수({len(mdd_predictions)})와 섹터 목록 개수({len(sectors)})가 다릅니다.")
            print("   'train.py'의 OUTPUT_DIM과 'predict.py'의 OUTPUT_DIM이 동일한지 확인하세요.")
        else:
            for sector_name, mdd_value in zip(sectors, mdd_predictions):
                print(f" - {sector_name:<30} : {mdd_value:.2f} %")

        # (이 mdd_predictions 리스트를 DB에 저장하거나
        #  API 응답으로 프론트엔드에 전송하면 됩니다.)

    else:
        print("\n❌ 최종 MDD 예측에 실패했습니다.")