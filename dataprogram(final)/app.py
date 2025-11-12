import pandas as pd
import numpy as np
from pykrx import stock
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from flask import Flask, render_template, request, jsonify
from io import BytesIO
import base64

# ---------------------------------------------------
# 📌 필수 해결책: Matplotlib 백엔드를 Agg로 설정 (macOS 스레드 오류 방지)
plt.switch_backend('Agg')
# ---------------------------------------------------

# --- Flask 앱 초기화 ---
app = Flask(__name__)

# --- 설정값 및 상수 ---
# ⚠️ CSV 파일 경로를 Flask 앱 루트 폴더 기준으로 설정 (app.py와 같은 위치)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISASTER_CSV_PATH = os.path.join(BASE_DIR, 'event.csv')
ANALYSIS_MONTHS_BEFORE = 3
ANALYSIS_MONTHS_AFTER = 3

# KOSPI 200 Big Sector Index Codes
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

# --- 1. 분석 함수 정의 ---


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

# --- 2. CSV 파일 로드 및 재난 일시 검색 함수 ---


def get_all_disaster_names():
    """CSV 파일에서 재난명 목록과 날짜를 로드합니다."""
    try:
        try:
            df_disaster = pd.read_csv(DISASTER_CSV_PATH, encoding='euc-kr')
        except UnicodeDecodeError:
            df_disaster = pd.read_csv(DISASTER_CSV_PATH, encoding='utf-8')
    except FileNotFoundError:
        raise Exception(f"CSV 파일을 찾을 수 없습니다: {DISASTER_CSV_PATH}")
    except Exception:
        raise Exception("CSV 파일 로드 중 오류가 발생했습니다.")

    if 'EventName' not in df_disaster.columns:
        raise ValueError("CSV 파일에 'EventName' 컬럼이 없습니다.")
    if 'EventDate' not in df_disaster.columns:
        raise ValueError("CSV 파일에 'EventDate' 컬럼이 없습니다.")

    return df_disaster[['EventName', 'EventDate']].drop_duplicates()


def get_event_date_by_name(disaster_name: str, df_events: pd.DataFrame) -> str:
    """로드된 DataFrame에서 특정 재난의 날짜를 찾습니다."""
    result = df_events[df_events['EventName'] == disaster_name]
    if result.empty:
        raise ValueError(f"'{disaster_name}'에 해당하는 재난을 찾을 수 없습니다.")

    event_datetime_str = result['EventDate'].iloc[0]
    return pd.to_datetime(event_datetime_str).strftime('%Y-%m-%d')


# --- 3. 핵심 분석 실행 로직 (Base64 인코딩 포함) ---
def run_analysis_for_event(event_name):
    """특정 재난명에 대한 섹터 분석을 실행하고 결과를 DataFrame과 Base64 차트로 반환합니다."""

    df_events = get_all_disaster_names()

    try:
        # 이벤트 날짜 및 기간 설정
        EVENT_DATE_STR = get_event_date_by_name(event_name, df_events)
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

    # 데이터 수집 및 분석 실행
    for sector_name, code in SECTOR_CODES.items():
        try:
            df_ohlcv = stock.get_index_ohlcv_by_date(
                start_date, end_date, code)

            if df_ohlcv.empty:
                continue

            # pykrx의 '종가'를 'Close'로 변경하여 기존 함수와 호환되도록 함
            df_ohlcv.rename(columns={'종가': 'Close'}, inplace=True)

            analysis = analyze_sector_performance(df_ohlcv['Close'])

            # 시각화를 위한 정규화 데이터 저장
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

    # MDD 순으로 정렬
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

        # Trough 및 Peak 로직
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

        # MDD 시각화 요소
        plt.scatter(peak_date, peak_price_norm, color='green',
                    marker='o', s=80, label='Peak')
        plt.scatter(trough_date, prices_norm.loc[trough_date],
                    color='red', marker='o', s=80, label='Trough')
        plt.annotate(f'MDD: {mdd_value}%', (trough_date, prices_norm.loc[trough_date]), textcoords="offset points", xytext=(
            0, -20), ha='center', fontsize=11, color='red', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6))

        plt.title(
            f'Price Movement: {sector_name} (MDD: {mdd_value}%)', fontsize=14)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Normalized Index (Start=100)', fontsize=10)
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
        'event_date': EVENT_DATE_STR,
        'start_date': start_date,
        'end_date': end_date,
        'combined_chart': combined_chart_base64,
        'individual_charts': individual_charts
    }


# --- 6. Flask 라우팅 설정 ---

@app.route('/', methods=['GET'])
def index():
    try:
        df_events = get_all_disaster_names()
        disasters = list(df_events['EventName'].unique())
    except Exception as e:
        return render_template('index.html', error=str(e), disasters=[])

    return render_template('index.html', disasters=disasters)


@app.route('/analyze', methods=['POST'])
def analyze_web_data():
    event_name = request.form.get('event_name')
    if not event_name:
        return jsonify({'error': '재난명을 선택해주세요.'}), 400

    analysis_data = run_analysis_for_event(event_name)

    if 'error' in analysis_data:
        return jsonify(analysis_data), 400

    # HTML 테이블로 변환
    df_html = analysis_data['df_results'].set_index('Sector Name')
    table_html = df_html.to_html(
        classes='table table-striped', float_format='%.2f')

    # 웹페이지에 필요한 데이터 전체를 반환 (Base64 차트 데이터 포함)
    return jsonify({
        'table_html': table_html,
        'event_name': analysis_data['event_name'],
        'event_date': analysis_data['event_date'],
        'start_date': analysis_data['start_date'],
        'end_date': analysis_data['end_date'],
        'combined_chart_base64': analysis_data['combined_chart'],
        'individual_charts': analysis_data['individual_charts']
    })


@app.route('/api/analyze/<event_name>', methods=['GET'])
def api_analyze_data(event_name):
    """
    외부 시스템을 위한 JSON API 엔드포인트 (Base64 차트 데이터 포함)
    """
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
    app.run(debug=True)
