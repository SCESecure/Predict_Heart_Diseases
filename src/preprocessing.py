# 라이브러리 불러오기

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

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

print("상태 : 데이터 셋 불러오기 완료")

# 두 feature와 target을 합침

row_total_data = pd.concat([row_feature_data, row_target_data], axis=1)

print(row_total_data.head())
print("상태 : 데이터 셋 합치기 완료")

# 데이터 셋에서 필요한 column들
data_col = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'num']
feature_col = data_col[0:5]
target_col = data_col[-1]

print(feature_col, target_col)

# feature_data = row_feature_data[feature_data_col]
# target_data = row_target_data[target_data_col]

# train 데이터 셋과 test 데이터 셋 분리
# 시드? 비율?
X_train, X_test, y_train, y_test = train_test_split(row_total_data[feature_col], row_total_data[target_col], test_size=0.3, random_state=216)

# Scaling
scaler = StandardScaler()

# train 데이터 스케일링
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_col)

print(X_train_scaled.head())
print("train 데이터 스케일링 완료")

# test 데이터 스케일링
X_test_scaled = scaler.transform(X_test)
y_test_scaled = scaler.transform(y_test)

print(X_test_scaled.head())
print(y_test_scaled.head())
print("test 데이터 스케일링 완료")
