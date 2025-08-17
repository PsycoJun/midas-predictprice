import pandas as pd
from prophet import Prophet
import joblib
from fastapi import FastAPI
from datetime import datetime, timedelta
import os
import requests
import numpy as np

# --- Global Settings & Configuration ---
app = FastAPI()

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# API & Model Parameters
KAMIS_API_KEY = '282e859d-5dac-4f18-9ffc-5557da39980e'
KAMIS_API_ID = '6042'
RETRAINING_THRESHOLD_MAPE = 15.0 # 재훈련을 트리거할 MAPE 임계값 (%)
OPTIMAL_CPS_VALUES = [0.001, 0.01, 0.05, 0.1, 0.5] # 테스트할 changepoint_prior_scale 값들

# Crop-specific configurations
CROP_CONFIGS = {
    "lettuce": {
        "item_code": "212",  # 청상추 품목코드
        "kind_code": "00",   # 청상추 품종코드 (해당없음)
        "data_filename": "lettuce_price_data.csv",
        "model_filename": "lettuce_model.joblib",
        "rank_display": "상품",
        "unit_display": "kg"
    },
    "spinach": {
        "item_code": "213",  # 시금치 품목코드
        "kind_code": "00",   # 시금치 품종코드
        "data_filename": "spinach_price_data.csv",
        "model_filename": "spinach_model.joblib",
        "rank_display": "상품",
        "unit_display": "kg"
    }
}

# --- Helper Functions (Data Fetching & Preprocessing) ---

def fetch_kamis_data(start_day, end_day, item_code, kind_code):
    url = 'http://www.kamis.or.kr/service/price/xml.do'
    params = {
        'action': 'periodProductList', 'p_cert_key': KAMIS_API_KEY,
        'p_cert_id': KAMIS_API_ID, 'p_returntype': 'json',
        'p_productclscode': '01', 'p_startday': start_day,
        'p_endday': end_day, 'p_itemcategorycode': '200', # 엽채류
        'p_itemcode': item_code, 'p_kindcode': kind_code,
        'p_productrankcode': '04', 'p_countrycode': '1101', # 서울
        'p_convert_kg_yn': 'Y'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('data', {}).get('item', [])
    except Exception as e:
        print(f"[ERROR] Failed to fetch KAMIS data for item {item_code}: {e}")
        return []

def format_and_clean_data(data):
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df = df[df['price'] != '-'].copy()
    df['ds'] = pd.to_datetime(df['yyyy'].astype(str) + '/' + df['regday'])
    df['y'] = df['price'].str.replace(',', '').astype(int)
    return df[['ds', 'y']].groupby('ds').mean().reset_index()

def find_best_model_params(df, recent_actuals):
    """최적의 changepoint_prior_scale을 찾아 모델을 훈련합니다."""
    best_mape = float('inf')
    best_cps = None
    best_model = None

    print("Finding best changepoint_prior_scale...")
    for cps in OPTIMAL_CPS_VALUES:
        try:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative', changepoint_prior_scale=cps)
            model.add_country_holidays(country_name='KR')
            model.fit(df)

            # 최근 데이터에 대한 예측 및 MAPE 계산
            future_df = model.make_future_dataframe(periods=0)
            future_df = future_df[future_df['ds'].isin(recent_actuals['ds'])]
            if not future_df.empty:
                forecast = model.predict(future_df)
                eval_df = pd.merge(recent_actuals, forecast[['ds', 'yhat']], on='ds')
                mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / eval_df['y'])) * 100
                print(f"  CPS: {cps}, MAPE: {mape:.2f}%")

                if mape < best_mape:
                    best_mape = mape
                    best_cps = cps
                    best_model = model
            else:
                print(f"  CPS: {cps}, No recent actuals to evaluate.")
        except Exception as e:
            print(f"  Error with CPS {cps}: {e}")
            continue

    if best_model is None:
        print("Could not find a suitable model. Using default CPS 0.05.")
        best_cps = 0.05
        best_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative', changepoint_prior_scale=best_cps)
        best_model.add_country_holidays(country_name='KR')
        best_model.fit(df)

    print(f"Best changepoint_prior_scale found: {best_cps} (MAPE: {best_mape:.2f}%)")
    return best_model

# --- Core Model & API Logic ---

@app.get("/predict/{crop_name}")
def predict_price(crop_name: str):
    """Main API endpoint to get the 30-day lettuce price forecast."""
    
    crop_name = crop_name.lower()
    if crop_name not in CROP_CONFIGS:
        return {"error": f"Crop '{crop_name}' not supported. Supported crops are: {', '.join(CROP_CONFIGS.keys())}"}

    config = CROP_CONFIGS[crop_name]
    # 파일 경로를 SCRIPT_DIR 기준으로 설정
    data_file = os.path.join(SCRIPT_DIR, config["data_filename"])
    model_file = os.path.join(SCRIPT_DIR, config["model_filename"])
    item_code = config["item_code"]
    kind_code = config["kind_code"]

    # 1. Update local data if necessary
    today = datetime.today()
    # 데이터 파일이 존재하면 읽어오고, 없으면 초기 3년치 데이터 가져오기
    if os.path.exists(data_file):
        df = pd.read_csv(data_file).rename(columns={'date': 'ds', 'price': 'y'})
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df.dropna(subset=['y'], inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])
        
        last_date_in_file = df['ds'].max()
        # 파일의 마지막 날짜가 오늘보다 이전이면, 그 다음 날부터 오늘까지의 데이터 업데이트
        if last_date_in_file < today:
            print(f"Data for {crop_name} is not up-to-date. Fetching data from {last_date_in_file + timedelta(days=1)} to today...")
            new_raw_data = fetch_kamis_data((last_date_in_file + timedelta(days=1)).strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'), item_code, kind_code)
            new_df = format_and_clean_data(new_raw_data)
            if not new_df.empty:
                df = pd.concat([df, new_df]).drop_duplicates(subset=['ds'], keep='last').sort_values('ds')
                df.to_csv(data_file, index=False)
                print(f"Data for {crop_name} updated successfully.")
            else:
                print(f"No new data found for {crop_name} from {last_date_in_file + timedelta(days=1)} to today.")
        else:
            print(f"Data for {crop_name} is already up-to-date.")
    else:
        print(f"No local data found for {crop_name}. Fetching last 3 years...")
        raw_data = fetch_kamis_data((today - timedelta(days=3*365)).strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'), item_code, kind_code)
        df = format_and_clean_data(raw_data)
        if not df.empty:
            df.to_csv(data_file, index=False)
            print(f"Initial 3-year data for {crop_name} fetched and saved.")
        else:
            print(f"Failed to fetch initial 3-year data for {crop_name}.")

    # 2. Load previous model or train a new one
    model = None
    retrain_needed = False
    recent_actuals = pd.DataFrame() # recent_actuals 초기화

    if os.path.exists(model_file):
        print(f"Loading existing model for {crop_name}...")
        model = joblib.load(model_file)
        
        # 3. Evaluate and potentially retrain the model
        # Get the last 7 days of actuals to compare with forecast
        recent_actuals = df[df['ds'] > df['ds'].max() - timedelta(days=7)]
        if not recent_actuals.empty:
            future_df = model.make_future_dataframe(periods=0)
            future_df = future_df[future_df['ds'].isin(recent_actuals['ds'])]
            if not future_df.empty:
                forecast = model.predict(future_df)
                # Calculate MAPE
                eval_df = pd.merge(recent_actuals, forecast[['ds', 'yhat']], on='ds')
                mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / eval_df['y'])) * 100
                print(f"Recent 7-day MAPE for {crop_name}: {mape:.2f}%.")

                if mape > RETRAINING_THRESHOLD_MAPE:
                    print(f"MAPE {mape:.2f}% exceeds threshold {RETRAINING_THRESHOLD_MAPE}%. Retraining model for {crop_name}...")
                    retrain_needed = True

    if model is None or retrain_needed:
        print(f"Training new model or retraining for {crop_name}...")
        model = find_best_model_params(df, recent_actuals) # 최적 파라미터 찾기 및 모델 훈련
        joblib.dump(model, model_file) # Save the newly trained model
        print(f"Model training complete and saved for {crop_name}.")

    # 4. Make the final prediction for -7 days to +30 days
    future = model.make_future_dataframe(periods=30)
    # Filter to include last 7 days of historical data and next 30 days of forecast
    forecast_range_start = df['ds'].max() - timedelta(days=7)
    forecast_result = model.predict(future)
    forecast_result = forecast_result[forecast_result['ds'] >= forecast_range_start]

    # Convert forecast to a list of dictionaries for API response
    detailed_forecast = []
    for index, row in forecast_result.iterrows():
        detailed_forecast.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "price": round(row['yhat'])
        })

    # 5. Extract and return the required information
    # Highest predicted price in the *future 30 days*
    future_30_days_forecast = forecast_result[forecast_result['ds'] > df['ds'].max()]
    highest_price_day = future_30_days_forecast.loc[future_30_days_forecast['yhat'].idxmax()]
    average_price = future_30_days_forecast['yhat'].mean()

    return {
        "crop": crop_name,
        "rank": config["rank_display"],
        "unit": config["unit_display"],
        "highest_predicted_price": {
            "date": highest_price_day['ds'].strftime('%Y-%m-%d'),
            "price": round(highest_price_day['yhat'])
        },
        "30_day_average_price": round(average_price),
        "detailed_forecast": detailed_forecast
    }

# To run the server, use the command:
# uvicorn api_server:app --reload
