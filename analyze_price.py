import pandas as pd

# 분석할 파일 이름
FILE_NAME = 'lettuce_price_data.csv'

def analyze_data(file_path):
    """CSV 파일을 읽어 기본적인 데이터 분석을 수행합니다."""
    try:
        # CSV 파일을 DataFrame으로 읽기
        df = pd.read_csv(file_path)
        print(f"--- '{file_path}' 파일 정보 ---")

        # 1. 데이터 구조 확인 (처음 5줄)
        print("\n[1. 데이터 샘플 (처음 5줄)]")
        print(df.head())

        # 2. 기본 통계 정보
        print("\n[2. 가격 데이터 기본 통계]")
        # describe()는 숫자형 컬럼에 대해서만 동작하므로 price 컬럼을 명시
        if 'price' in df.columns:
            print(df['price'].describe())
        else:
            print("'price' 컬럼을 찾을 수 없습니다.")

        # 3. 데이터 컬럼 정보 (데이터 타입, 결측치 등)
        print("\n[3. 데이터프레임 정보]")
        df.info()

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"데이터 분석 중 오류 발생: {e}")

if __name__ == '__main__':
    analyze_data(FILE_NAME)
