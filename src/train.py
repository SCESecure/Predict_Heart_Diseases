# 전처리 했던 변수들 가져옴
from preprocessing import X_train_scaled, y_train, X_test_scaled, y_test

# 앙상블로 특성 선택에 필요한 라이브러리
from sklearn.feature_selection import SelectFromModel

# 모델들
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# 정확도 평가
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# 그 이외의 라이브러리
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

print("선택된 특성들에 대한 데이터는 다음과 같습니다.")
print("[X_train_scaled (이하 X_train_selected)]")
X_train_selected = X_train_scaled.loc[:, selected_features]
print(X_train_selected)

print("[X_test_scaled (이하 X_test_selected)]")
X_test_selected = X_test_scaled.loc[:, selected_features]
print(X_test_selected)

print("\n")

# Logistic Regression 학습
LR_model = LogisticRegression()
LR_model.fit(X_train_selected, y_train)
LR_pred = LR_model.predict(X_test_selected)
LR_pred_prob = LR_model.predict_proba(X_test_selected)

LR_acc = accuracy_score(y_test, LR_pred)
LR_roc_auc_ovo = roc_auc_score(y_test, LR_pred_prob, multi_class='ovo')
LR_roc_auc_ovr = roc_auc_score(y_test, LR_pred_prob, multi_class='ovr')

print("\n")
print("Logistic Regression 모델 학습 완료")
print("Acc : " + str(LR_acc) + " ({:.2f}%)".format(LR_acc * 100))
print("ROC AUC 점수(OVO(one-vs-one)) : " + str(LR_roc_auc_ovo) + " ({:.2f}%)".format(LR_roc_auc_ovo * 100))
print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(LR_roc_auc_ovr) + " ({:.2f}%)".format(LR_roc_auc_ovr * 100))

print("\n")

# SVM(Support Vector Machine) 학습
SVM_kernel = ['linear', 'poly', 'rbf', 'sigmoid'] # precomputed 커널은 데이터가 정방행렬만 가능
SVM_acc_list = []
SVM_roc_acc_ovo_list = []
SVM_roc_acc_ovr_list = []

for i in range(0, len(SVM_kernel)) :
    # SVC의 probality 속성이 scikit-learn 1.11 버전부터 deprecated 됨에 따라
    # CalibratedClassifierCV() 함수로 이를 대체함

    if (i == 'poly') :
        SVM_model = CalibratedClassifierCV(SVC(kernel=SVM_kernel[i], C=5, degree=3, random_state=1), ensemble=False)
    else :
        SVM_model = CalibratedClassifierCV(SVC(kernel=SVM_kernel[i], C=5, random_state=1), ensemble=False)
    
    SVM_model.fit(X_train_selected, y_train)

    SVM_pred = SVM_model.predict(X_test_selected)
    SVM_pred_prob = SVM_model.predict_proba(X_test_selected)

    SVM_acc = accuracy_score(y_test, SVM_pred)
    SVM_roc_acc_ovo = roc_auc_score(y_test, SVM_pred_prob, multi_class='ovo')
    SVM_roc_acc_ovr = roc_auc_score(y_test, SVM_pred_prob, multi_class='ovr')

    SVM_acc_list.insert(i, SVM_acc)
    SVM_roc_acc_ovo_list.insert(i, SVM_roc_acc_ovo)
    SVM_roc_acc_ovr_list.insert(i, SVM_roc_acc_ovr)

print("SVM(Support Vector Machine) 모델 학습 완료")

for i in range(0, len(SVM_kernel)) :
    print("[" + SVM_kernel[i] + " 모델] ")
    print("Acc : " + str(SVM_acc_list[i]) + " ({:.2f}%)".format(SVM_acc_list[i] * 100))
    print("ROC AUC 점수(OVO(one-vs-one)) : " + str(SVM_roc_acc_ovo_list[i]) + " ({:.2f}%)".format(SVM_roc_acc_ovo_list[i] * 100))
    print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(SVM_roc_acc_ovr_list[i]) + " ({:.2f}%)".format(SVM_roc_acc_ovr_list[i] * 100))
    print("")

print("")

# Random Forest 학습 (앙상블 방법에서 사용된 모델 가져옴)
RF_model = forest
RF_model.fit(X_train_selected, y_train)

RF_pred = RF_model.predict(X_test_selected)
RF_pred_prob = RF_model.predict_proba(X_test_selected)

RF_acc = accuracy_score(y_test, RF_pred)
RF_roc_auc_ovo = roc_auc_score(y_test, RF_pred_prob, multi_class='ovo')
RF_roc_auc_ovr = roc_auc_score(y_test, RF_pred_prob, multi_class='ovr')

print("Random Forest 모델 학습 완료")
print("Acc : " + str(RF_acc) + " ({:.2f}%)".format(RF_acc * 100))
print("ROC AUC 점수(OVO(one-vs-one)) : " + str(RF_roc_auc_ovo) + " ({:.2f}%)".format(RF_roc_auc_ovo * 100))
print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(RF_roc_auc_ovr) + " ({:.2f}%)".format(RF_roc_auc_ovr * 100))

print("\n")

# KNN(K-Nearest Neighbors) 학습
KNN_model = KNeighborsClassifier(n_neighbors=10)
KNN_model.fit(X_train_selected, y_train)

KNN_pred = KNN_model.predict(X_test_selected)
KNN_pred_prob = KNN_model.predict_proba(X_test_selected)

KNN_acc = accuracy_score(y_test, KNN_pred)
KNN_roc_auc_ovo = roc_auc_score(y_test, KNN_pred_prob, multi_class='ovo')
KNN_roc_auc_ovr = roc_auc_score(y_test, KNN_pred_prob, multi_class='ovr')

print("KNN(K-Nearest Neighbors) 모델 학습 완료")
print("Acc : " + str(KNN_acc) + " ({:.2f}%)".format(KNN_acc * 100))
print("ROC AUC 점수(OVO(one-vs-one)) : " + str(KNN_roc_auc_ovo) + " ({:.2f}%)".format(KNN_roc_auc_ovo * 100))
print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(KNN_roc_auc_ovr) + " ({:.2f}%)".format(KNN_roc_auc_ovr * 100))

print("\n")

# XGBoost 학습 (설정은 Random Forest와 똑같이 하였음)
XGB_model = XGBClassifier(n_estimators=100, max_depth=5, random_state=1)
XGB_model.fit(X_train_selected, y_train)

XGB_pred = XGB_model.predict(X_test_selected)
XGB_pred_prob = XGB_model.predict_proba(X_test_selected)

XGB_acc = accuracy_score(y_test, XGB_pred)
XGB_roc_auc_ovo = roc_auc_score(y_test, XGB_pred_prob, multi_class='ovo')
XGB_roc_auc_ovr = roc_auc_score(y_test, XGB_pred_prob, multi_class='ovr')

print("XGBoost (eXtra Gradient Boost) 분류기 모델 학습 완료")
print("Acc : " + str(XGB_acc) + " ({:.2f}%)".format(XGB_acc * 100))
print("ROC AUC 점수(OVO(one-vs-one)) : " + str(XGB_roc_auc_ovo) + " ({:.2f}%)".format(XGB_roc_auc_ovo * 100))
print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(XGB_roc_auc_ovr) + " ({:.2f}%)".format(XGB_roc_auc_ovr * 100))
