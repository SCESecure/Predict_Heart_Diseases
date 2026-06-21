# Predict_Heart_Diseases(CardioCare)

## 개요

- 이 모델은 UCI에서 제공하는 UCI Heart Disease(CardioCare) 데이터 셋을 사용한 모델입니다.
- 데이터 셋에서는 특정 타겟의 나이, 성별, 흉통, 혈압(휴식기), 콜레스테롤, 공복혈당(FBS), 안정시 심전도(Resting ECG), 최대 혈압 수치, 운동협심증, ST 분절 수치, 심혈관 직경에 따른 심장병 등에 대한 데이터를 가지고 있습니다. (더 자세한 건 [Heart Disease(UCI)](https://archive.ics.uci.edu/dataset/45/heart+disease) 확인 바랍니다.)

## 실행 방법

### 1. 준비 단계

- 자신의 로컬 PC에 git과 python(3.13 버전)을 설치하여 주십시오.
- 해당 repository의 URL을 복사한 다음 자신의 로컬 PC에서 `git clone` 하시면 파일을 손쉽게 가져오실 수 있습니다.

### 2. 의존성 설치 단계

- 프로젝트에서 `pip install -r requirements.txt` 를 하여 의존성 설치 바랍니다.
- 가능하다면 python 가상 환경(venv)를 만들고 설치하시는 것을 권장합니다.
  - 프로젝트에서 `python -m venv .venv` 를 사용하여 가상 환경 구축
  - MacOS 및 Linux : `source .venv/bin/activate` 를 사용하여 가상 환경 활성화
  - Windows : `.\.venv\Scripts\activate` 를 사용하여 가상 환경 활성화
  - `deactivate` 를 사용하여 가상 환경 비활성화

### 3. 모델 훈련 단계

- `python ./src/train.py` 를 사용하여 모델을 훈련시키길 바랍니다.

### 4. Docker 이미지 빌드

- `docker build -t cardiocare:1.0 .` 과 같이 docker 이미지를 빌드하시길 바랍니다.

### 5. 테스팅

- 테스팅의 경우 unittest를 사용하여 진행합니다. `python -m unittest tests/test_pipeline.py` 를 사용하여 구축된 테스트 파이프라인을 실행 바랍니다.
