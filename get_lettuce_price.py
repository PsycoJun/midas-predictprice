
import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# --- 설정 ---
# 사용자의 KAMIS API 키와 ID
API_KEY = '282e859d-5dac-4f18-9ffc-5557da39980e'
API_ID = '6042'

# 조회할 품목 정보 (청상추, 상품)
ITEM_CATEGORY_CODE = '200' # 부류코드: 엽채류
ITEM_CODE = '212'          # 품목코드: 청상추
KIND_CODE = '00'           # 종류코드: 해당없음
RANK_CODE = '04'           # 등급코드: 상품

# 저장할 파일 이름
OUTPUT_FILE_NAME = 'lettuce_price_data.csv'

# --- 함수 정의 ---

def fetch_kamis_data(start_day, end_day):
    """KAMIS API를 호출하여 지정된 기간의 농산물 가격 데이터를 가져옵니다."""
    url = 'http://www.kamis.or.kr/service/price/xml.do'
    
    params = {
        'action': 'periodProductList',
        'p_cert_key': API_KEY,
        'p_cert_id': API_ID,
        'p_returntype': 'json',
        'p_productclscode': '01', # 구분: 소매
        'p_startday': start_day,
        'p_endday': end_day,
        'p_itemcategorycode': ITEM_CATEGORY_CODE,
        'p_itemcode': ITEM_CODE,
        'p_kindcode': KIND_CODE,
        'p_productrankcode': RANK_CODE,
        'p_countrycode': '1101', # 도매법인코드: 서울
        'p_convert_kg_yn': 'Y'   # kg 단위 환산여부
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and 'item' in data['data']:
            return data['data']['item']
        else:
            if 'error_code' in data:
                print(f"API Error: {data.get('error_code')} - {data.get('message', 'Unknown error')}")
            else:
                print("API 응답에 데이터가 없습니다. 응답 구조를 확인하세요.")
                print("Received data:", data)
            return None
    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
        return None
    except ValueError:
        print("JSON 데이터를 파싱하는 데 실패했습니다. 응답 내용을 확인하세요.")
        print("Raw response:", response.text)
        return None

def format_data(data):
    """가져온 원시 데이터를 pandas DataFrame으로 변환하고 형식을 정리합니다."""
    if not data or len(data) == 0:
        return None

    df = pd.DataFrame(data)
    
    columns_to_keep = ['itemname', 'kindname', 'rank', 'yyyy', 'regday', 'price']
    df_selected = df[[col for col in columns_to_keep if col in df.columns]]

    if 'yyyy' in df_selected.columns and 'regday' in df_selected.columns:
        df_selected['date'] = df_selected['yyyy'].astype(str) + '/' + df_selected['regday']
        df_selected['date'] = pd.to_datetime(df_selected['date'], format='%Y/%m/%d')
        df_selected = df_selected.drop(columns=['yyyy', 'regday'])
        
        # 가격에서 쉼표 제거하고 숫자로 변환
        if 'price' in df_selected.columns:
            # 가격이 '-'인 행(결측치)은 제외
            df_selected = df_selected[df_selected['price'] != '-'].copy()
            # 쉼표 제거 및 정수형으로 변환
            df_selected['price'] = df_selected['price'].str.replace(',', '').astype(int)

        cols = ['date'] + [col for col in df_selected.columns if col != 'date']
        df_selected = df_selected[cols]
        return df_selected
    else:
        return None

def main():
    """메인 실행 함수: 파일 존재 여부에 따라 데이터 다운로드 또는 업데이트를 수행합니다."""
    today = datetime.today()

    if os.path.exists(OUTPUT_FILE_NAME):
        print(f"'{OUTPUT_FILE_NAME}' 파일을 찾았습니다. 데이터 업데이트를 확인합니다.")
        try:
            existing_df = pd.read_csv(OUTPUT_FILE_NAME)
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            last_date = existing_df['date'].max()
            
            if (today - last_date).days >= 7:
                print("데이터가 7일 이상 경과하여 최신 7일치 데이터를 추가합니다.")
                start_update_date = today - timedelta(days=7)
                start_day_str = start_update_date.strftime('%Y-%m-%d')
                end_day_str = today.strftime('%Y-%m-%d')

                new_raw_data = fetch_kamis_data(start_day_str, end_day_str)
                if new_raw_data:
                    new_df = format_data(new_raw_data)
                    if new_df is not None:
                        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['date'], keep='last')
                        combined_df = combined_df.sort_values(by='date', ascending=False)
                        combined_df.to_csv(OUTPUT_FILE_NAME, index=False, encoding='utf-8-sig')
                        print(f"데이터를 성공적으로 업데이트했습니다. '{OUTPUT_FILE_NAME}'")
                    else:
                        print("새로운 데이터를 포맷하는 데 실패했습니다.")
                else:
                    print("새로운 데이터를 가져오지 못했습니다.")
            else:
                print("데이터가 최신 상태입니다 (7일 이내).")
        except Exception as e:
            print(f"기존 파일 처리 중 오류 발생: {e}. 전체 데이터를 새로 받습니다.")
            os.remove(OUTPUT_FILE_NAME) # 문제가 있는 파일은 삭제
            main() # 재귀 호출로 전체 데이터 다운로드 로직 실행

    else:
        print(f"'{OUTPUT_FILE_NAME}' 파일을 찾을 수 없습니다. 최근 3년치 데이터를 1년 단위로 나누어 다운로드합니다.")
        all_data_df = pd.DataFrame()
        
        for i in range(3):
            end_date = today - timedelta(days=i*365)
            start_date = today - timedelta(days=(i+1)*365)
            
            start_day_str = start_date.strftime('%Y-%m-%d')
            end_day_str = end_date.strftime('%Y-%m-%d')
            
            print(f"{i+1}번째 데이터 수집 중: {start_day_str} ~ {end_day_str}")
            raw_data_part = fetch_kamis_data(start_day_str, end_day_str)
            
            if raw_data_part:
                df_part = format_data(raw_data_part)
                if df_part is not None:
                    all_data_df = pd.concat([all_data_df, df_part])

        if not all_data_df.empty:
            all_data_df = all_data_df.drop_duplicates(subset=['date']).sort_values(by='date', ascending=False)
            all_data_df.to_csv(OUTPUT_FILE_NAME, index=False, encoding='utf-8-sig')
            print(f"총 {len(all_data_df)}개의 데이터를 성공적으로 '{OUTPUT_FILE_NAME}' 파일에 저장했습니다.")
        else:
            print("데이터를 수집하지 못했습니다.")


if __name__ == '__main__':
    main()
