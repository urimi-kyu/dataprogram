import urllib.request
import json
import datetime
import numpy as np
import sys
import urllib.parse
import mdd_predictor  # 👈 mdd_predictor.py 임포트

# --- 1. Naver API 및 상수 ---

# (중요!) Naver API 인증 정보 (⚠️ 실제 값으로 대체해야 합니다.)
CLIENT_ID = "xAkr8e4sWLnQW4x_YnIh"
CLIENT_SECRET = "gkDg6FlbBR"

# 모델이 사용하는 8개 재난 키워드 순서 (모델 입력 차원 8과 일치)
MODEL_INPUT_ORDER = [
    "산불", "지진", "태풍", "감염병", "가뭄", "폭설", "홍수", "기타"
]
# 크롤링 시 홍수와 함께 체크할 키워드
DISASTERS_TO_CHECK = MODEL_INPUT_ORDER + ["폭우"]

# 모델이 사용하는 12개 섹터 이름 (MDD 예측 출력 차원 12와 일치)
SECTORS = [
    "코스피 전체 시장",            # Market (KOSPI)
    "코스피200 통신서비스",        # Communication Services
    "코스피200 건설",              # Construction
    "코스피200 중공업",            # Heavy Industry
    "코스피200 철강·소재",         # Steel/Materials
    "코스피200 에너지·화학",       # Energy/Chemicals
    "코스피200 정보기술",          # Information Technology
    "코스피200 금융",              # Finance
    "코스피200 필수소비재",        # Consumer Staples
    "코스피200 자유소비재",        # Consumer Discretionary
    "코스피200 산업재",            # Industrials
    "코스피200 헬스케어"           # Healthcare
]

# FE 뉴스 가져오기
def fetch_disaster_news(keyword, max_results=3):
    """특정 재난 키워드에 대한 최신 뉴스 리스트(title, link) 반환"""
    TODAY_DATE_STR = datetime.date.today().strftime("%Y%m%d")

    encText = urllib.parse.quote(f"{keyword} 재난 피해")

    url = (
        f"https://openapi.naver.com/v1/search/news.json?query={encText}"
        f"&display={max_results}&sort=sim&start=1"
        f"&enddate={TODAY_DATE_STR}&startdate={TODAY_DATE_STR}"
    )

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", CLIENT_SECRET)

    try:
        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if rescode != 200:
            return []

        response_body = response.read()
        result = json.loads(response_body.decode('utf-8'))

        items = result.get("items", [])
        news_list = []
        for item in items:
            news_list.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "description": item.get("description", "")
            })
        return news_list

    except Exception:
        return []

def get_today_disaster_news_grouped(max_results=3):
    """MODEL_INPUT_ORDER 기준 8개 재난별 뉴스 리스트 반환"""
    grouped = {}
    for disaster in MODEL_INPUT_ORDER:
        grouped[disaster] = fetch_disaster_news(disaster, max_results=max_results)
    return grouped
# FE END

# --- 2. 뉴스 크롤링 및 벡터 변환 함수 ---


def search_disaster_occurrence_news(keyword):
    """네이버 뉴스 API를 호출하여 특정 키워드의 오늘 뉴스 건수를 반환합니다."""
    TODAY_DATE_STR = datetime.date.today().strftime("%Y%m%d")

    encText = urllib.parse.quote(f"'{keyword}' AND (속보 OR 발생 OR 피해)")

    url = (f"https://openapi.naver.com/v1/search/news.json?query={encText}"
           f"&display=100&sort=sim&start=1&enddate={TODAY_DATE_STR}&startdate={TODAY_DATE_STR}")

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", CLIENT_SECRET)

    try:
        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if rescode == 200:
            response_body = response.read()
            result = json.loads(response_body.decode('utf-8'))
            return result.get('total', 0)
        else:
            print(f"❌ [{keyword}] API Error Code: {rescode}", file=sys.stderr)
            return 0
    except urllib.error.URLError as e:
        print(f"❌ [{keyword}] URL Error: {e.reason}. 인증 정보 확인 필요.",
              file=sys.stderr)
        return 0
    except Exception as e:
        print(f"❌ [{keyword}] Unknown Error: {e}", file=sys.stderr)
        return 0


def convert_counts_to_vector(disaster_counts: dict) -> list:
    """크롤링된 뉴스 건수를 모델 입력 벡터(8차원)로 변환합니다."""
    vector = []

    # '홍수' 및 '폭우' 카운트를 합산
    flood_count = disaster_counts.get("홍수", 0) + disaster_counts.get("폭우", 0)

    # MODEL_INPUT_ORDER 순서에 맞춰 벡터 생성
    for disaster in MODEL_INPUT_ORDER:
        if disaster == "홍수":
            count = flood_count
        else:
            count = disaster_counts.get(disaster, 0)

        # 정규화 (모델 학습 시 사용한 정규화 방식과 일치해야 함)
        intensity = min(1.0, count / 100.0)
        vector.append(intensity)

    return vector


# --- 3. 메인 오케스트레이션 함수 (app.py가 호출) ---

def get_today_mdd_prediction(main_keyword):
    """오늘 날짜의 뉴스 강도를 기반으로 MDD 예측을 수행하고 결과를 반환합니다."""

    # 1. 뉴스 크롤링 실행
    disaster_counts = {}
    for disaster in DISASTERS_TO_CHECK:
        count = search_disaster_occurrence_news(disaster)
        disaster_counts[disaster] = count

    # 2. 크롤링 결과를 AI 모델 입력 벡터(점수)로 변환
    disaster_vector_list = convert_counts_to_vector(disaster_counts)

    # 모델 입력 형태: (1, 8) numpy.float32
    feature_vector = np.array(disaster_vector_list).reshape(
        1, mdd_predictor.INPUT_DIM).astype(np.float32)

    # 3. AI 모델 호출하여 MDD 예측
    try:
        mdd_predictions_vector = mdd_predictor.predict_mdd_value(
            feature_vector)
    except RuntimeError as e:
        return {'error': f"MDD 모델 예측 실행 실패: {e}"}

    # 4. 최종 결과 정리
    if mdd_predictions_vector is None or len(mdd_predictions_vector) != len(SECTORS):
        return {'error': "MDD 모델 예측 결과의 차원이 일치하지 않습니다. 모델 파일을 확인하세요."}

    # 가장 높은 MDD 값과 그 섹터를 찾음 (최악의 시나리오)
    max_mdd_index = np.argmax(mdd_predictions_vector)
    max_mdd_value = mdd_predictions_vector[max_mdd_index]
    max_mdd_sector = SECTORS[max_mdd_index]

    # 상세 결과 문자열 구성
    detail_results = [
        f"{SECTORS[i]}: {mdd_predictions_vector[i]:.2f}%" for i in range(len(SECTORS))]
    detail_text = "전 섹터 예측 MDD: " + ", ".join(detail_results)
    
    # 데이터를 리스트 형태로 가공
    sector_data = []
    for i, sector_name in enumerate(SECTORS):
        value = mdd_predictions_vector[i]
        sector_data.append({
            'name': sector_name,
            'value': float(value)
        })

    # 오늘 날짜를 기준으로 예측했다고 표시
    event_date_str = datetime.date.today().strftime("%Y-%m-%d")

    return {
        'status': 'success',
        'event_name': main_keyword,
        'event_date': event_date_str,
        'predicted_mdd': f"{max_mdd_value:.2f}% ({max_mdd_sector})",
        'detail': detail_text,
        'sector_data': sector_data
    }
