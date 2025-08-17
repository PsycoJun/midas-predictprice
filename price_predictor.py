import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- 설정 ---
INPUT_FILE = 'lettuce_price_data.csv'
OUTPUT_PREDICTION_FILE = 'prediction.csv'
OUTPUT_PLOT_FILE = 'prediction_plot.png'
PREDICTION_DAYS = 30 # 예측할 기간 (일)

# --- 1. 데이터 전처리 ---
def preprocess_data(file_path):
    """데이터를 읽고 Prophet 모델에 맞게 전처리합니다."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return None

    # itemname 필터링 로직이 모든 데이터를 삭제하는 문제를 해결하기 위해 해당 로직을 제거합니다.
    # if 'itemname' in df.columns:
    #     df = df[df['itemname'] != '[]']

    # 날짜 형식 변환 및 Prophet 형식에 맞게 컬럼명 변경
    df['ds'] = pd.to_datetime(df['date'])
    df = df[['ds', 'price']].rename(columns={'price': 'y'})

    # 중복된 날짜 데이터 처리 (일자별 평균값 사용)
    df = df.groupby('ds').mean().reset_index()
    
    print("데이터 전처리 완료. 처리된 데이터 수:", len(df))
    return df

# --- 2. 모델 훈련 및 예측 ---
def train_and_predict(df):
    """Prophet 모델을 훈련하고 미래를 예측합니다."""
    # Prophet 모델 생성 및 훈련
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode='multiplicative', # 가격 데이터는 보통 곱셈 모델이 더 적합
                    changepoint_prior_scale=0.1) 
    
    # 대한민국 공휴일 추가
    model.add_country_holidays(country_name='KR')
    
    print("모델 훈련 시작...")
    model.fit(df)
    print("모델 훈련 완료.")

    # 미래 예측 데이터프레임 생성 및 예측
    future = model.make_future_dataframe(periods=PREDICTION_DAYS)
    forecast = model.predict(future)
    
    print(f"{PREDICTION_DAYS}일 후의 가격 예측 완료.")
    return model, forecast

# --- 3. 성능 평가 ---
def evaluate_model(model):
    """교차 검증을 통해 모델의 성능을 평가합니다."""
    print("모델 성능 평가 시작 (교차 검증)...")
    # 데이터 양에 맞춰 교차 검증 파라미터 조정
    initial_days = '180 days'
    period_days = '90 days'
    horizon_days = '30 days'
    
    try:
        df_cv = cross_validation(model, initial=initial_days, period=period_days, horizon=horizon_days)
        
        # 성능 지표 계산
        df_p = performance_metrics(df_cv)
        
        print("모델 성능 평가 완료.")
        # MAPE(Mean Absolute Percentage Error) 값을 백분율로 변환하여 출력
        mape = df_p['mape'].values[0] * 100
        print(f"\n[모델 성능 평가 결과]")
        print(f"- 예측 정확도 (1 - MAPE): {100 - mape:.2f}%")
        print(f"- 평균 절대 백분율 오차 (MAPE): {mape:.2f}%")
        print("  (MAPE는 0에 가까울수록 좋으며, 예측값이 실제값에서 평균적으로 몇 % 벗어나는지를 의미합니다.)")
        print(df_p.head())
    except ValueError as e:
        print(f"\n교차 검증 중 오류 발생: {e}")
        print("데이터 양이 충분하지 않아 교차 검증을 건너뜁니다. 예측은 정상적으로 수행됩니다.")


# --- 4. 결과 저장 및 시각화 ---
def save_results(model, forecast):
    """예측 결과와 그래프를 파일로 저장합니다."""
    # 예측 결과 CSV 파일로 저장
    prediction_df = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].tail(PREDICTION_DAYS)
    prediction_df.to_csv(OUTPUT_PREDICTION_FILE)
    print(f"\n예측 결과가 '{OUTPUT_PREDICTION_FILE}' 파일에 저장되었습니다.")
    print(prediction_df)

    # 예측 결과 그래프 저장
    # 과거 데이터는 제외하고 예측 기간만 플로팅
    fig = model.plot(forecast, xlabel='Date', ylabel='Price (KRW)')
    ax = fig.gca()
    ax.set_title('Lettuce Price Forecast (Next 30 Days)', fontsize=16)
    # x축 범위를 예측 시작일로부터 30일 후까지로 제한
    ax.set_xlim(pd.to_datetime(forecast['ds'].iloc[-PREDICTION_DAYS]), forecast['ds'].max())
    fig.savefig(OUTPUT_PLOT_FILE)
    print(f"Prediction plot saved to '{OUTPUT_PLOT_FILE}'")
    plt.close()


if __name__ == '__main__':
    # 1. 데이터 로드 및 전처리
    processed_df = preprocess_data(INPUT_FILE)
    
    if processed_df is not None and not processed_df.empty and len(processed_df) > 2:
        # 2. 모델 훈련 및 예측
        model, forecast = train_and_predict(processed_df)
        
        # 3. 모델 성능 평가
        evaluate_model(model)
        
        # 4. 결과 저장
        save_results(model, forecast)
    else:
        print("모델 훈련에 필요한 데이터가 부족하여 예측을 중단합니다.")
