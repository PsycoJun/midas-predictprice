# 농산물 가격 예측 API

이 프로젝트는 KAMIS (농산물유통정보) API를 사용하여 특정 농산물(상추, 시금치)의 과거 가격 데이터를 수집하고, Prophet 모델을 사용하여 미래 30일간의 가격을 예측하는 API 서버를 제공합니다.

## 주요 기능

- **자동 데이터 수집 및 업데이트**: KAMIS API를 통해 최신 가격 데이터를 주기적으로 가져와 로컬 데이터를 업데이트합니다.
- **동적 모델 재훈련**: 예측 모델의 성능(MAPE)을 지속적으로 평가하여, 정확도가 일정 수준 이하로 떨어지면 자동으로 모델을 재훈련합니다.
- **다중 작물 지원**: 상추(lettuce)와 시금치(spinach) 두 가지 작물에 대한 가격 예측을 지원합니다.
- **RESTful API**: FastAPI를 사용하여 구축된 깔끔하고 사용하기 쉬운 API 엔드포인트를 제공합니다.

## 프로젝트 구조

```
midas/
│
├── api_server.py           # 메인 FastAPI 서버
├── price_predictor.py      # 독립 실행 가능한 가격 예측 스크립트 (상추 전용)
├── get_lettuce_price.py    # 독립 실행 가능한 데이터 수집 스크립트 (상추 전용)
├── analyze_price.py        # 데이터 분석 유틸리티 스크립트
│
├── lettuce_price_data.csv  # 상추 가격 데이터 파일
├── spinach_price_data.csv  # 시금치 가격 데이터 파일
│
├── lettuce_model.joblib    # 저장된 상추 가격 예측 모델
└── spinach_model.joblib    # 저장된 시금치 가격 예측 모델
```

### 파일 설명

- **`api_server.py`**: 프로젝트의 핵심 파일입니다. FastAPI를 사용하여 API 서버를 실행합니다. 데이터 수집, 모델 훈련/재훈련, 예측 등 모든 주요 로직이 포함되어 있습니다.
- **`price_predictor.py`**: `api_server.py`의 예측 기능을 단순화하여 독립적으로 실행할 수 있는 스크립트입니다. `lettuce_price_data.csv`를 읽어 예측을 수행하고 `prediction.csv`와 `prediction_plot.png`를 생성합니다.
- **`get_lettuce_price.py`**: 상추 가격 데이터를 KAMIS API로부터 가져와 `lettuce_price_data.csv` 파일로 저장하는 독립적인 스크립트입니다.
- **`analyze_price.py`**: 데이터 파일(`lettuce_price_data.csv`)의 기본적인 통계 정보를 출력하는 유틸리티 스크립트입니다.
- **`*.csv`**: 각 작물의 가격 데이터가 저장된 파일입니다. `api_server` 실행 시 자동으로 생성 및 업데이트될 수 있습니다.
- **`*.joblib`**: 훈련된 Prophet 모델이 직렬화되어 저장된 파일입니다. `api_server` 실행 시 자동으로 생성 및 업데이트될 수 있습니다.

## 설치 및 실행

### 1. 필요 라이브러리 설치

프로젝트 실행에 필요한 파이썬 라이브러리들을 설치합니다.

```bash
pip install pandas prophet fastapi "uvicorn[standard]" joblib requests numpy
```

### 2. API 서버 실행

API 서버를 실행하는 것이 이 프로젝트의 주된 사용 방법입니다.

1.  `midas` 폴더로 이동합니다.
    ```bash
    cd path/to/midas
    ```
2.  `uvicorn`을 사용하여 서버를 실행합니다.
    ```bash
    uvicorn api_server:app --reload
    ```
3.  서버가 실행되면 `http://127.0.0.1:8000` 주소로 접속할 수 있습니다.

### 3. API 사용법

서버가 실행된 후, 웹 브라우저나 `curl`과 같은 도구를 사용하여 예측 결과를 요청할 수 있습니다.

- **상추(lettuce) 가격 예측:**
  ```bash
  curl http://127.0.0.1:8000/predict/lettuce
  ```
- **시금치(spinach) 가격 예측:**
  ```bash
  curl http://127.0.0.1:8000/predict/spinach
  ```

### 4. 독립 스크립트 실행 (선택 사항)

- **데이터 수동 업데이트 (`get_lettuce_price.py`):**
  ```bash
  python get_lettuce_price.py
  ```
- **단순 가격 예측 (`price_predictor.py`):**
  ```bash
  python price_predictor.py
  ```

---
