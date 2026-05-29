# 불러오는 코드의 경우 https://archive.ics.uci.edu/dataset/45/heart+disease 에서 ucimlrepo를 통해 가져옴
# 데이터는 processed.cleveland.data에 해당


from ucimlrepo import fetch_ucirepo 

import pandas as pd
  
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

