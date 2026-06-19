from train import X_test_selected, y_test, \
    LR_model, SVM_kernel, SVM_fitted_list, RF_model, KNN_model, XGB_model

from sklearn.metrics import accuracy_score, roc_auc_score

print("\n\n")
print("[Inference 단계 (inference.py)]")
print("\n\n")

# Logistic Regression 모델 예측 및 점수
print("Logistic Regression 모델에 대하여 예측합니다.")
LR_pred = LR_model.predict(X_test_selected)
LR_pred_prob = LR_model.predict_proba(X_test_selected)

LR_acc = accuracy_score(y_test, LR_pred)
LR_roc_auc_ovo = roc_auc_score(y_test, LR_pred_prob, multi_class='ovo')
LR_roc_auc_ovr = roc_auc_score(y_test, LR_pred_prob, multi_class='ovr')

print("정확도 점수는 다음과 같습니다.")
print("Acc : " + str(LR_acc) + " ({:.2f}%)".format(LR_acc * 100))
print("ROC AUC 점수(OVO(one-vs-one)) : " + str(LR_roc_auc_ovo) + " ({:.2f}%)".format(LR_roc_auc_ovo * 100))
print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(LR_roc_auc_ovr) + " ({:.2f}%)".format(LR_roc_auc_ovr * 100))

print("\n")

# SVM(Support Vector Machine) 모델 예측 및 점수
SVM_acc_list = []
SVM_roc_acc_ovo_list = []
SVM_roc_acc_ovr_list = []

print("SVM(Support Vector Machine) 모델에 대하여 예측합니다.")
for i in range(0, len(SVM_fitted_list)) :

    SVM_pred = SVM_fitted_list[i].predict(X_test_selected)
    SVM_pred_prob = SVM_fitted_list[i].predict_proba(X_test_selected)

    SVM_acc = accuracy_score(y_test, SVM_pred)
    SVM_roc_acc_ovo = roc_auc_score(y_test, SVM_pred_prob, multi_class='ovo')
    SVM_roc_acc_ovr = roc_auc_score(y_test, SVM_pred_prob, multi_class='ovr')

    SVM_acc_list.insert(i, SVM_acc)
    SVM_roc_acc_ovo_list.insert(i, SVM_roc_acc_ovo)
    SVM_roc_acc_ovr_list.insert(i, SVM_roc_acc_ovr)

print("정확도 점수는 다음과 같습니다.")
for i in range(0, len(SVM_kernel)) :
    print("[" + SVM_kernel[i] + " 모델] ")
    print("Acc : " + str(SVM_acc_list[i]) + " ({:.2f}%)".format(SVM_acc_list[i] * 100))
    print("ROC AUC 점수(OVO(one-vs-one)) : " + str(SVM_roc_acc_ovo_list[i]) + " ({:.2f}%)".format(SVM_roc_acc_ovo_list[i] * 100))
    print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(SVM_roc_acc_ovr_list[i]) + " ({:.2f}%)".format(SVM_roc_acc_ovr_list[i] * 100))
    print("")

print("")

# Random Forest 모델 예측 및 점수
print("Random Forest 모델에 대하여 예측합니다.")
RF_pred = RF_model.predict(X_test_selected)
RF_pred_prob = RF_model.predict_proba(X_test_selected)

RF_acc = accuracy_score(y_test, RF_pred)
RF_roc_auc_ovo = roc_auc_score(y_test, RF_pred_prob, multi_class='ovo')
RF_roc_auc_ovr = roc_auc_score(y_test, RF_pred_prob, multi_class='ovr')

print("정확도 점수는 다음과 같습니다.")
print("Acc : " + str(RF_acc) + " ({:.2f}%)".format(RF_acc * 100))
print("ROC AUC 점수(OVO(one-vs-one)) : " + str(RF_roc_auc_ovo) + " ({:.2f}%)".format(RF_roc_auc_ovo * 100))
print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(RF_roc_auc_ovr) + " ({:.2f}%)".format(RF_roc_auc_ovr * 100))

print("\n")

# KNN(K-Nearest Neighbors) 모델 예측 및 점수
print("KNN(K-Nearest Neighbors) 모델에 대하여 예측합니다.")
KNN_pred = KNN_model.predict(X_test_selected)
KNN_pred_prob = KNN_model.predict_proba(X_test_selected)

KNN_acc = accuracy_score(y_test, KNN_pred)
KNN_roc_auc_ovo = roc_auc_score(y_test, KNN_pred_prob, multi_class='ovo')
KNN_roc_auc_ovr = roc_auc_score(y_test, KNN_pred_prob, multi_class='ovr')

print("정확도 점수는 다음과 같습니다.")
print("Acc : " + str(KNN_acc) + " ({:.2f}%)".format(KNN_acc * 100))
print("ROC AUC 점수(OVO(one-vs-one)) : " + str(KNN_roc_auc_ovo) + " ({:.2f}%)".format(KNN_roc_auc_ovo * 100))
print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(KNN_roc_auc_ovr) + " ({:.2f}%)".format(KNN_roc_auc_ovr * 100))

print("\n")

# XGBoost(eXtra Gradient Boost) 모델 예측 및 점수
print("XGBoost(eXtra Gradient Boost) 모델에 대하여 예측합니다.")
XGB_pred = XGB_model.predict(X_test_selected)
XGB_pred_prob = XGB_model.predict_proba(X_test_selected)

XGB_acc = accuracy_score(y_test, XGB_pred)
XGB_roc_auc_ovo = roc_auc_score(y_test, XGB_pred_prob, multi_class='ovo')
XGB_roc_auc_ovr = roc_auc_score(y_test, XGB_pred_prob, multi_class='ovr')

print("정확도 점수는 다음과 같습니다.")
print("Acc : " + str(XGB_acc) + " ({:.2f}%)".format(XGB_acc * 100))
print("ROC AUC 점수(OVO(one-vs-one)) : " + str(XGB_roc_auc_ovo) + " ({:.2f}%)".format(XGB_roc_auc_ovo * 100))
print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(XGB_roc_auc_ovr) + " ({:.2f}%)".format(XGB_roc_auc_ovr * 100))

print("\n")