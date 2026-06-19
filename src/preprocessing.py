# 라이브러리 불러오기

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

print("\n\n")
print("[Preprocessing 단계 (preprocessing.py)]")
print("\n\n")

# 데이터 셋 불러오기
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# metadata 
# print(heart_disease.metadata) 
  
# variable information 
# print(heart_disease.variables) 

# feature_df = pd.read_table(X)

X.to_csv('./data/heart_disease_feature.csv')
y.to_csv('./data/heart_disease_target.csv')

feature_file_path = "./data/heart_disease_feature.csv"
target_file_path = "./data/heart_disease_target.csv"

row_feature_data = pd.read_csv(feature_file_path)
row_target_data = pd.read_csv(target_file_path)

print("데이터 셋 불러오기 완료")
print("[row_feature_data]")
print(row_feature_data.head())
print("[row_target_data]")
print(row_target_data.head())

print("\n")

# 두 feature와 target을 합침
row_total_data = pd.concat([row_feature_data, row_target_data], axis=1)

print("데이터 셋 합치기 완료")
print("[row_total_data]")
print(row_total_data.head())

print("\n")

# 데이터 셋에서 필요한 column들
data_col = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'num']
feature_col = data_col[0:5]
target_col = data_col[-1]

print("데이터 셋에 필요한 특성들은 다음과 같습니다")
print(feature_col, target_col)

# feature_data = row_feature_data[feature_data_col]
# target_data = row_target_data[target_data_col]

print("\n")

# train 데이터 셋과 test 데이터 셋 분리
# 시드? 비율?
X_train, X_test, y_train, y_test = train_test_split(row_total_data[feature_col], row_total_data[target_col], test_size=0.3, random_state=1)

# Scaling
scaler = StandardScaler()

# train 데이터 스케일링
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_col)

print("train 데이터 스케일링 완료")
print(X_train_scaled.head())

print("\n")

# test 데이터 스케일링
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_col)

print("test 데이터 스케일링 완료")
print(X_test_scaled.head())

print("\n")

X_train_scaled.to_csv('./data/X_train_scaled_data.csv')
X_test_scaled.to_csv('./data/X_test_scaled_data.csv')
print("X_train과 X_test 데이터가 csv파일로 정상적으로 내보내기 완료했습니다.")
