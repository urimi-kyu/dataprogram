import pandas as pd
import numpy as np
from pykrx import stock
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from flask import Flask, render_template, request, jsonify
from io import BytesIO
import base64
from urllib.parse import unquote
import sys
import mdd_prediction_service  # 📌 MDD 예측 서비스 모듈 임포트

# ---------------------------------------------------
plt.switch_backend('Agg')
# ---------------------------------------------------

# --- Flask 앱 초기화 ---
app = Flask(__name__)

# --- 설정값 및 상수 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISASTER_CSV_PATH = os.path.join(BASE_DIR, 'event.csv')
ANALYSIS_MONTHS_BEFORE = 3
ANALYSIS_MONTHS_AFTER = 3

# KOSPI 200 Big Sector Index Codes (총 12개)
SECTOR_CODES = {
    'Market (KOSPI)': '1001',
    'KOSPI 200 - Communication Services': '1150',
    'KOSPI 200 - Construction': '1151',
    'KOSPI 200 - Heavy Industry': '1152',
    'KOSPI 200 - Steel/Materials': '1153',
    'KOSPI 200 - Energy/Chemicals': '1154',
    'KOSPI 200 - Information Technology': '1155',
    'KOSPI 200 - Finance': '1156',
    'KOSPI 200 - Consumer Staples': '1157',
    'KOSPI 200 - Consumer Discretionary': '1158',
    'KOSPI 200 - Industrials': '1159',
    'KOSPI 200 - Healthcare': '1160'
}

# --- 1. 분석 유틸리티 함수 정의 ---


def analyze_sector_performance(close_prices: pd.Series) -> dict:
    """주가 데이터(종가 Series)를 받아 MDD, 최종 증감률, Final Drawdown을 계산합니다."""
    rolling_max = close_prices.expanding().max()
    drawdown = (close_prices / rolling_max) - 1
    mdd_value = round(drawdown.min() * -1 * 100, 2)
    cumulative_return = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
    cumulative_return_pct = round(cumulative_return * 100, 2)
    final_dd_pct = round(drawdown.iloc[-1] * 100, 2)

    return {
        'MDD (%)': mdd_value,
        'Final Return (%)': cumulative_return_pct,
        'Final DD (%)': final_dd_pct,
        'drawdown_series': drawdown
    }


# --- 2. CSV 파일 로드 및 검색 함수 ---
EXPECTED_COLUMNS = ['EventName', 'EventDate', 'EventType']


def load_csv_with_encoding_fallback():
    """여러 인코딩을 시도하고, 실패 시 헤더를 수동 복구하여 DataFrame을 로드합니다."""
    df_disaster = None

    for encoding in ['utf-8', 'euc-kr', 'cp949']:
        try:
            df_disaster = pd.read_csv(DISASTER_CSV_PATH, encoding=encoding)
            df_disaster.encoding = encoding
            break
        except Exception:
            continue

    if df_disaster is None:
        raise Exception("CSV 파일 로드 실패: 지원되지 않는 인코딩이거나 파일이 손상되었습니다.")

    if not all(col in df_disaster.columns for col in EXPECTED_COLUMNS):

        try:
            df_disaster_noheader = pd.read_csv(
                DISASTER_CSV_PATH, header=None, encoding=df_disaster.encoding)

            header_row = df_disaster_noheader.iloc[0]
            df_disaster = df_disaster_noheader[1:].copy()
            df_disaster.columns = header_row.tolist()

            if df_disaster.columns.tolist()[:len(EXPECTED_COLUMNS)] != EXPECTED_COLUMNS:
                df_disaster.columns = EXPECTED_COLUMNS + \
                    df_disaster.columns.tolist()[len(EXPECTED_COLUMNS):]

        except Exception as e:
            raise ValueError(
                f"필수 컬럼 [{', '.join(EXPECTED_COLUMNS)}]을 찾을 수 없습니다. 현재 컬럼: {df_disaster.columns.tolist()}")

    return df_disaster[EXPECTED_COLUMNS].drop_duplicates()


def get_disaster_types_and_events():
    """CSV 파일에서 모든 재난 종류(EventType)와 이벤트를 로드합니다."""
    df_disaster = load_csv_with_encoding_fallback()

    disaster_types = sorted(df_disaster['EventType'].unique().tolist())
    events_by_type = df_disaster.groupby('EventType')['EventName'].unique().apply(
        lambda x: sorted(x.tolist())).to_dict()

    return disaster_types, events_by_type


def get_all_disaster_names():
    """CSV 파일에서 모든 재난명을 평면 리스트로 로드합니다."""
    df_disaster = load_csv_with_encoding_fallback()

    return df_disaster[['EventName']].drop_duplicates()


def get_event_date_by_name(disaster_name: str) -> str:
    """특정 재난의 날짜를 찾습니다."""
    df_disaster = load_csv_with_encoding_fallback()

    result = df_disaster[df_disaster['EventName'] == disaster_name]
    if result.empty:
        raise ValueError(f"'{disaster_name}'에 해당하는 재난을 찾을 수 없습니다.")

    event_datetime_str = result['EventDate'].iloc[0]
    return pd.to_datetime(event_datetime_str).strftime('%Y-%m-%d')


def find_stock_code(corp_name):
    """회사 이름(종목명)으로 6자리 종목 코드를 찾습니다."""
    try:
        df_list = stock.get_market_ticker_list()
        for ticker in df_list:
            name = stock.get_market_ticker_name(ticker)
            if name == corp_name:
                return ticker
        return None
    except Exception:
        return None

# --- 3. 핵심 분석 실행 로직 (섹터 전체) ---


def run_analysis_for_event(event_name):
    """특정 재난명에 대한 섹터 분석을 실행하고 결과를 DataFrame과 Base64 차트로 반환합니다."""

    try:
        EVENT_DATE_STR = get_event_date_by_name(event_name)
        event_date_dt = pd.to_datetime(EVENT_DATE_STR)

        start_date = (
            event_date_dt - pd.DateOffset(months=ANALYSIS_MONTHS_BEFORE)).strftime('%Y%m%d')
        end_date = (
            event_date_dt + pd.DateOffset(months=ANALYSIS_MONTHS_AFTER)).strftime('%Y%m%d')

    except Exception as e:
        return {'error': str(e)}

    results_list = []
    all_sector_prices = {}
    all_sector_analysis = {}

    for sector_name, code in SECTOR_CODES.items():
        try:
            df_ohlcv = stock.get_index_ohlcv_by_date(
                start_date, end_date, code)

            if df_ohlcv.empty:
                continue

            df_ohlcv.rename(columns={'종가': 'Close'}, inplace=True)

            analysis = analyze_sector_performance(df_ohlcv['Close'])

            start_price = df_ohlcv['Close'].iloc[0]
            prices_norm = (df_ohlcv['Close'] / start_price) * 100
            all_sector_prices[sector_name] = prices_norm
            all_sector_analysis[sector_name] = {
                'prices': df_ohlcv['Close'], 'mdd_info': analysis}

            results_list.append({
                'Sector Name': sector_name,
                'MDD (%)': analysis['MDD (%)'],
                'Final DD (%)': analysis['Final DD (%)'],
                'Final Return (%)': analysis['Final Return (%)'],
            })

        except Exception:
            continue

    df_results = pd.DataFrame(results_list)

    if df_results.empty:
        return {'error': "분석 기간 동안 유효한 주식 데이터가 없습니다."}

    df_results = df_results.sort_values(by='MDD (%)', ascending=False)

    # --- 4. 그래프 생성 및 Base64 인코딩 로직 ---

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    # 4-1. 통합 비교 차트 생성 (Base64)
    plt.figure(figsize=(15, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    for sector_name, prices in all_sector_prices.items():
        plt.plot(prices.index, prices.values, label=sector_name)
    plt.title(
        f'Sector Price Movement Comparison (Normalized: Start=100) \n Event Date: {EVENT_DATE_STR}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Stock Index', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Sector')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()
    plt.axvline(x=event_date_dt, color='red', linestyle='-',
                linewidth=2, label='Event Date')
    plt.axhline(y=100, color='gray', linestyle='--', linewidth=1)
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    combined_chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 4-2. 개별 섹터 차트 생성 (Base64 리스트)
    individual_charts = []
    for sector_name, analysis_data in all_sector_analysis.items():
        prices_norm = all_sector_prices[sector_name]
        prices_raw = analysis_data['prices']
        full_analysis = analyze_sector_performance(prices_raw)
        mdd_value = full_analysis['MDD (%)']
        drawdown_series = full_analysis['drawdown_series']

        trough_date = drawdown_series.idxmin()
        rolling_max_raw = prices_raw.expanding().max()
        peak_price_raw_at_trough = rolling_max_raw.loc[trough_date]
        potential_peak_dates = rolling_max_raw.index[(rolling_max_raw.index <= trough_date) & (
            rolling_max_raw == peak_price_raw_at_trough)]
        peak_date = potential_peak_dates.max()
        peak_price_norm = prices_norm.loc[peak_date]

        # 차트 그리기
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.plot(prices_norm.index, prices_norm.values,
                 label=sector_name, color='blue')
        plt.scatter(peak_date, peak_price_norm, color='green',
                    marker='o', s=80, label='Peak')
        plt.scatter(trough_date, prices_norm.loc[trough_date],
                    color='red', marker='o', s=80, label='Trough')
        plt.annotate(f'MDD: {mdd_value}%', (trough_date, prices_norm.loc[trough_date]), textcoords="offset points", xytext=(
            0, -20), ha='center', fontsize=11, color='red', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6))
        plt.title(
            f'Price Movement: {sector_name} (MDD: {mdd_value}%)', fontsize=14)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Normalized Stock Index (Start=100)', fontsize=10)
        plt.axvline(x=event_date_dt, color='gray', linestyle='--',
                    linewidth=1, label='Event Date')
        plt.axhline(y=100, color='gray', linestyle=':', linewidth=1)
        plt.legend(loc='upper left')
        plt.grid(True)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        individual_charts.append({
            'name': sector_name,
            'base64_img': base64.b64encode(buffer.getvalue()).decode('utf-8')
        })

    # --- 5. 최종 반환 ---
    return {
        'df_results': df_results,
        'event_name': event_name,
        'event_date': event_date_dt.strftime('%Y-%m-%d'),
        'start_date': start_date,
        'end_date': end_date,
        'combined_chart': combined_chart_base64,
        'individual_charts': individual_charts
    }

# --- 6. 개별 기업 분석 로직 ---


def analyze_individual_stock(event_name, corp_name, months_before, months_after):
    """개별 종목의 MDD와 차트를 분석합니다."""

    try:
        event_date_str = get_event_date_by_name(event_name)
        event_date_dt = pd.to_datetime(event_date_str)

        ticker = find_stock_code(corp_name)

        if not ticker:
            raise ValueError(f"'{corp_name}' ticker not found.")

        start_date = (event_date_dt -
                      pd.DateOffset(months=months_before)).strftime('%Y%m%d')
        end_date = (event_date_dt +
                    pd.DateOffset(months=months_after)).strftime('%Y%m%d')

        df_ohlcv = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)

        if df_ohlcv.empty:
            raise ValueError(
                f"No data available for '{corp_name}' in the period.")

        df_ohlcv.rename(columns={'종가': 'Close'}, inplace=True)

        analysis = analyze_sector_performance(df_ohlcv['Close'])

        # --- 차트 생성 로직 ---
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.unicode_minus'] = False

        prices_raw = df_ohlcv['Close']
        prices_norm = (prices_raw / prices_raw.iloc[0]) * 100

        full_analysis = analyze_sector_performance(prices_raw)
        mdd_value = full_analysis['MDD (%)']
        drawdown_series = full_analysis['drawdown_series']

        trough_date = drawdown_series.idxmin()
        rolling_max_raw = prices_raw.expanding().max()
        peak_price_raw_at_trough = rolling_max_raw.loc[trough_date]
        potential_peak_dates = rolling_max_raw.index[(rolling_max_raw.index <= trough_date) & (
            rolling_max_raw == peak_price_raw_at_trough)]
        peak_date = potential_peak_dates.max()
        peak_price_norm = prices_norm.loc[peak_date]

        # 차트 그리기
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')

        plt.plot(prices_norm.index, prices_norm.values,
                 label=f'{corp_name} ({ticker})', color='blue')
        plt.scatter(peak_date, peak_price_norm, color='green',
                    marker='o', s=80, label='Peak')
        plt.scatter(trough_date, prices_norm.loc[trough_date],
                    color='red', marker='o', s=80, label='Trough')

        plt.annotate(
            f'MDD: {mdd_value}%',
            (trough_date, prices_norm.loc[trough_date]),
            textcoords="offset points",
            xytext=(0, -20),
            ha='center',
            fontsize=11,
            color='red',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6)
        )

        plt.title(
            f'Price Movement: {corp_name} ({ticker}) | Event: {event_name}', fontsize=14)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Normalized Price (Start=100)', fontsize=10)
        plt.axvline(x=event_date_dt, color='gray', linestyle='--',
                    linewidth=1, label='Event Date')
        plt.axhline(y=100, color='gray', linestyle=':', linewidth=1)
        plt.legend(loc='upper left')
        plt.grid(True)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 결과 DataFrame 생성
        df_result = pd.DataFrame([analysis])
        df_result.rename(columns={'Final Return (%)': 'Return (%)',
                         'Final DD (%)': 'Final Drawdown (%)'}, inplace=True)
        df_result = df_result[['MDD (%)', 'Return (%)', 'Final Drawdown (%)']]
        df_result.index = [corp_name]

        return {
            'chart_base64': chart_base64,
            'table_html': df_result.to_html(classes='table table-striped', float_format='%.2f'),
            'mdd_value': mdd_value,
            'event_date': event_date_str,
            'corp_name': corp_name,
            'event_name': event_name
        }

    except Exception as e:
        return {'error': f"Data analysis failed: {str(e)}"}


# --- 7. Flask 라우팅 설정 ---

@app.route('/', methods=['GET'])
def index():
    try:
        _, events_by_type = get_disaster_types_and_events()
        disasters_flat = [event for sublist in events_by_type.values()
                          for event in sublist]
    except Exception as e:
        return render_template('index.html', error=str(e), disasters=[])

    return render_template('index.html', disasters=disasters_flat)


@app.route('/analyze', methods=['POST'])
def analyze_web_data():
    event_name = request.form.get('event_name')
    if not event_name:
        return jsonify({'error': '재난명을 선택해주세요.'}), 400

    analysis_data = run_analysis_for_event(event_name)

    if 'error' in analysis_data:
        return jsonify(analysis_data), 400

    df_html = analysis_data['df_results'].set_index('Sector Name')
    table_html = df_html.to_html(
        classes='table table-striped', float_format='%.2f')

    return jsonify({
        'table_html': table_html,
        'event_name': analysis_data['event_name'],
        'event_date': analysis_data['event_date'],
        'start_date': analysis_data['start_date'],
        'end_date': analysis_data['end_date'],
        'combined_chart_base64': analysis_data['combined_chart'],
        'individual_charts': analysis_data['individual_charts']
    })


@app.route('/individual_analysis', methods=['GET'])
def individual_analysis_page():
    try:
        disaster_types, _ = get_disaster_types_and_events()

    except Exception as e:
        return render_template('individual.html', error=str(e), disaster_types=[])

    return render_template('individual.html', disaster_types=disaster_types)


@app.route('/individual_analysis', methods=['POST'])
def handle_individual_analysis():
    event_name = request.form.get('event_name')
    corp_name = request.form.get('corp_name')

    if not event_name or not corp_name:
        return jsonify({'error': '재난 이름과 회사 이름을 모두 입력해주세요.'}), 400

    analysis_result = analyze_individual_stock(event_name, corp_name,
                                               ANALYSIS_MONTHS_BEFORE,
                                               ANALYSIS_MONTHS_AFTER)

    if 'error' in analysis_result:
        return jsonify(analysis_result), 400

    return jsonify({
        'status': 'success',
        'analysis_data': analysis_result
    })

# --- 8. MDD 예측 라우트 (오류 해결됨) ---


@app.route('/predict_mdd', methods=['GET'])
def predict_mdd_page():
    """MDD 예측 페이지 렌더링"""
    try:
        _, events_by_type = get_disaster_types_and_events()
        # 모든 재난명을 평탄화하여 드롭다운에 표시
        disasters_flat = [event for sublist in events_by_type.values()
                          for event in sublist]
    except Exception as e:
        return render_template('predict_mdd.html', error=str(e), disaster_types=[])

    return render_template('predict_mdd.html', disaster_types=disasters_flat)


@app.route('/predict_mdd', methods=['POST'])
def handle_mdd_prediction():
    """뉴스 크롤링 -> 벡터 생성 -> MDD 예측 실행"""

    event_name = request.form.get('event_name')

    if not event_name:
        return jsonify({'error': '재난명을 선택해주세요.'}), 400

    try:
        # **핵심: 외부 서비스 파일의 함수를 호출하여 모든 예측 로직 위임**
        # 📌 이 함수 이름이 mdd_prediction_service.py에서 정의한 이름과 일치해야 합니다.
        prediction_data = mdd_prediction_service.get_today_mdd_prediction(
            event_name)

        if 'error' in prediction_data:
            return jsonify({'error': prediction_data['error']}), 500

        # 성공 시, 결과 반환
        return jsonify(prediction_data)

    except Exception as e:
        print(f"MDD 예측 서비스 호출 중 치명적인 오류 발생: {e}", file=sys.stderr)
        return jsonify({'error': f"MDD 예측 서비스 호출 중 치명적인 오류 발생: {str(e)}"}), 500


# --- 9. API 엔드포인트 ---
@app.route('/api/types', methods=['GET'])
def get_disaster_types():
    """API: 재난 종류(EventType) 목록을 반환합니다."""
    try:
        disaster_types, _ = get_disaster_types_and_events()
        return jsonify({'types': disaster_types})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/events_by_type/<type_name>', methods=['GET'])
def get_events_by_type(type_name):
    """API: 선택된 종류(EventType)에 해당하는 재난 목록을 반환합니다."""
    try:
        _, events_by_type = get_disaster_types_and_events()

        type_name = unquote(type_name)

        events = events_by_type.get(type_name, [])
        return jsonify({'events': events})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/<event_name>', methods=['GET'])
def api_analyze_data(event_name):
    """API: 섹터 분석 결과를 JSON 형태로 반환합니다."""
    analysis_data = run_analysis_for_event(event_name)

    if 'error' in analysis_data:
        return jsonify(analysis_data), 400

    df_results = analysis_data['df_results'].reset_index()

    response_data = {
        'event_name': analysis_data['event_name'],
        'event_date': analysis_data['event_date'],
        'analysis_period': f"Start:{analysis_data['start_date']} End:{analysis_data['end_date']}",
        'sector_results': df_results.to_dict(orient='records'),
        'charts': {
            'combined_chart_base64': analysis_data['combined_chart'],
            'individual_charts': analysis_data['individual_charts']
        }
    }

    return jsonify(response_data)


if __name__ == '__main__':
    # 서버 시작 전 모델/스케일러 로드 시도
    try:
        # 📌 mdd_predictor.py에서 모델 로드 함수를 직접 호출 (mdd_prediction_service 내부에서 처리)
        # 이 시점에서 모듈 로드 실패 시, 에러가 발생하며 바로 알려줍니다.
        pass
    except Exception as e:
        # 이 부분은 현재 구조상 mdd_prediction_service.py 내부에서 처리됩니다.
        pass

    app.run(debug=True)
