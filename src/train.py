from preprocessing import X_train_scaled, y_train, X_test_scaled, y_test

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score

import pandas as pd

print("\n\n")
print("[Train 단계 (train.py)]")
print("\n\n")

# RandomForest 사용
forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
forest.fit(X_train_scaled, y_train)
print("RandomForest로 하여 X_train_scaled와 y_train fit 완료")
print("[X_train_scaled]")
print(X_train_scaled)
print("[y_train]")
print(y_train)

# 특성 선택
selector = SelectFromModel(forest, prefit=True)
features_bool = selector.get_support()
print("선택된 특성의 boolean 값 : " + str(features_bool))

selected_features = []
for i in range(0,X_train_scaled.shape[1]) :
    if (features_bool[i] == True) :
        selected_features.insert(i, X_train_scaled.columns[i])

# 선택된 특성 출력 (selected_features)
print("\n")
print("RandomForest에 의해 선택된 특성은 다음과 같습니다 : " + str(selected_features))

# Logistic Regression 학습
LR_model = LogisticRegression()
LR_model.fit(X_train_scaled, y_train)
LR_pred = LR_model.predict(X_test_scaled)
LR_pred_prob = LR_model.predict_proba(X_test_scaled)

LR_acc = accuracy_score(y_test, LR_pred)
LR_roc_auc = roc_auc_score(y_test, LR_pred_prob, multi_class='ovo')

print("\n")
print("Logistic Regression 모델 학습 완료")
print("Acc : " + str(LR_acc) + " ({:.2f}%)".format(LR_acc * 100))
print("ROC AUC 점수(OVO(one-vs-one)) : " + str(LR_roc_auc) + " ({:.2f}%)".format(LR_roc_auc * 100))

