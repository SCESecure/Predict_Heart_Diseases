# from train import X_test_selected, y_test, \
#     LR_model, SVM_kernel, SVM_fitted_list, RF_model, KNN_model, XGB_model

from sklearn.metrics import accuracy_score, roc_auc_score, \
                            precision_recall_fscore_support, balanced_accuracy_score, \
                            confusion_matrix

def LR_predict(LR_model, X_test_selected, y_test) :

    # Logistic Regression 모델 예측 및 점수
    print("Logistic Regression 모델에 대하여 예측합니다.")
    LR_pred = LR_model.predict(X_test_selected)
    LR_pred_prob = LR_model.predict_proba(X_test_selected)

    LR_acc = accuracy_score(y_test, LR_pred)
    LR_roc_auc_ovo = roc_auc_score(y_test, LR_pred_prob, multi_class='ovo')
    LR_roc_auc_ovr = roc_auc_score(y_test, LR_pred_prob, multi_class='ovr')
    LR_precision, LR_recall, LR_fscore, LR_support = precision_recall_fscore_support(y_test, LR_pred)
    LR_bacc = balanced_accuracy_score(y_test, LR_pred)

    LR_cm = confusion_matrix(y_test, LR_pred)

    print("모델 지표는 다음과 같습니다.")
    print("Acc : " + str(LR_acc) + " ({:.2f}%)".format(LR_acc * 100))
    print("ROC AUC 점수(OVO(one-vs-one)) : " + str(LR_roc_auc_ovo) + " ({:.2f}%)".format(LR_roc_auc_ovo * 100))
    print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(LR_roc_auc_ovr) + " ({:.2f}%)".format(LR_roc_auc_ovr * 100))
    print("Precision : " + str(LR_precision[0]))
    print("Recall : " + str(LR_recall[0]))
    print("F1-Score : " + str(LR_fscore[0]))
    print("Balanced Acc : " + str(LR_bacc))
    print("Confusion Matrix : \n", LR_cm)

    print("\n")

    total = [LR_acc, LR_roc_auc_ovo, LR_roc_auc_ovr, 
             LR_precision[0], LR_recall[0], LR_fscore[0],
             LR_bacc, LR_cm]

    return total


def SVM_predict(SVM_model, X_test_selected, y_test) :

    # SVM(Support Vector Machine) 모델 예측 및 점수
    print("SVM(Support Vector Machine) 모델에 대하여 예측합니다.")
    SVM_pred = SVM_model.predict(X_test_selected)
    SVM_pred_prob = SVM_model.predict_proba(X_test_selected)

    SVM_acc = accuracy_score(y_test, SVM_pred)
    SVM_roc_acc_ovo = roc_auc_score(y_test, SVM_pred_prob, multi_class='ovo')
    SVM_roc_acc_ovr = roc_auc_score(y_test, SVM_pred_prob, multi_class='ovr')
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, SVM_pred)
    bacc = balanced_accuracy_score(y_test, SVM_pred)
    
    cm = confusion_matrix(y_test, SVM_pred)

    print("모델 지표는 다음과 같습니다.")
    print("Acc : " + str(SVM_acc) + " ({:.2f}%)".format(SVM_acc * 100))
    print("ROC AUC 점수(OVO(one-vs-one)) : " + str(SVM_roc_acc_ovo) + " ({:.2f}%)".format(SVM_roc_acc_ovo * 100))
    print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(SVM_roc_acc_ovr) + " ({:.2f}%)".format(SVM_roc_acc_ovr * 100))
    print("Precision : " + str(precision[0]))
    print("Recall : " + str(recall[0]))
    print("F1-Score : " + str(fscore[0]))
    print("Balanced Acc : " + str(bacc))
    print("Confusion Matrix : \n", cm)
    print("\n")

    total = [SVM_acc, SVM_roc_acc_ovo, SVM_roc_acc_ovr,
            precision[0], recall[0], fscore[0],
            bacc, cm]

    return total


def RF_predict(RF_model, X_test_selected, y_test) :
    # Random Forest 모델 예측 및 점수
    print("Random Forest 모델에 대하여 예측합니다.")
    RF_pred = RF_model.predict(X_test_selected)
    RF_pred_prob = RF_model.predict_proba(X_test_selected)

    RF_acc = accuracy_score(y_test, RF_pred)
    RF_roc_auc_ovo = roc_auc_score(y_test, RF_pred_prob, multi_class='ovo')
    RF_roc_auc_ovr = roc_auc_score(y_test, RF_pred_prob, multi_class='ovr')
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, RF_pred)
    bacc = balanced_accuracy_score(y_test, RF_pred)
    
    cm = confusion_matrix(y_test, RF_pred)

    print("모델 지표는 다음과 같습니다.")
    print("Acc : " + str(RF_acc) + " ({:.2f}%)".format(RF_acc * 100))
    print("ROC AUC 점수(OVO(one-vs-one)) : " + str(RF_roc_auc_ovo) + " ({:.2f}%)".format(RF_roc_auc_ovo * 100))
    print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(RF_roc_auc_ovr) + " ({:.2f}%)".format(RF_roc_auc_ovr * 100))
    print("Precision : " + str(precision[0]))
    print("Recall : " + str(recall[0]))
    print("F1-Score : " + str(fscore[0]))
    print("Balanced Acc : " + str(bacc))
    print("Confusion Matrix : \n", cm)
    print("\n")

    total = [RF_acc, RF_roc_auc_ovo, RF_roc_auc_ovr,
            precision[0], recall[0], fscore[0],
            bacc, cm]

    return total

def KNN_predict(KNN_model, X_test_selected, y_test) :
    # KNN(K-Nearest Neighbors) 모델 예측 및 점수
    print("KNN(K-Nearest Neighbors) 모델에 대하여 예측합니다.")
    KNN_pred = KNN_model.predict(X_test_selected)
    KNN_pred_prob = KNN_model.predict_proba(X_test_selected)

    KNN_acc = accuracy_score(y_test, KNN_pred)
    KNN_roc_auc_ovo = roc_auc_score(y_test, KNN_pred_prob, multi_class='ovo')
    KNN_roc_auc_ovr = roc_auc_score(y_test, KNN_pred_prob, multi_class='ovr')
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, KNN_pred)
    bacc = balanced_accuracy_score(y_test, KNN_pred)
    
    cm = confusion_matrix(y_test, KNN_pred)

    print("모델 지표는 다음과 같습니다.")
    print("Acc : " + str(KNN_acc) + " ({:.2f}%)".format(KNN_acc * 100))
    print("ROC AUC 점수(OVO(one-vs-one)) : " + str(KNN_roc_auc_ovo) + " ({:.2f}%)".format(KNN_roc_auc_ovo * 100))
    print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(KNN_roc_auc_ovr) + " ({:.2f}%)".format(KNN_roc_auc_ovr * 100))
    print("Precision : " + str(precision[0]))
    print("Recall : " + str(recall[0]))
    print("F1-Score : " + str(fscore[0]))
    print("Balanced Acc : " + str(bacc))
    print("Confusion Matrix : \n", cm)
    print("\n")

    total = [KNN_acc, KNN_roc_auc_ovo, KNN_roc_auc_ovr,
            precision[0], recall[0], fscore[0],
            bacc, cm]

    return total


def XGB_predict(XGB_model, X_test_selected, y_test) :
    # XGBoost(eXtra Gradient Boost) 모델 예측 및 점수
    print("XGBoost(eXtra Gradient Boost) 모델에 대하여 예측합니다.")
    XGB_pred = XGB_model.predict(X_test_selected)
    XGB_pred_prob = XGB_model.predict_proba(X_test_selected)

    XGB_acc = accuracy_score(y_test, XGB_pred)
    XGB_roc_auc_ovo = roc_auc_score(y_test, XGB_pred_prob, multi_class='ovo')
    XGB_roc_auc_ovr = roc_auc_score(y_test, XGB_pred_prob, multi_class='ovr')
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, XGB_pred)
    bacc = balanced_accuracy_score(y_test, XGB_pred)
    
    cm = confusion_matrix(y_test, XGB_pred)

    print("모델 지표는 다음과 같습니다.")
    print("Acc : " + str(XGB_acc) + " ({:.2f}%)".format(XGB_acc * 100))
    print("ROC AUC 점수(OVO(one-vs-one)) : " + str(XGB_roc_auc_ovo) + " ({:.2f}%)".format(XGB_roc_auc_ovo * 100))
    print("ROC AUC 점수(OVR(one-vs-rest)) : " + str(XGB_roc_auc_ovr) + " ({:.2f}%)".format(XGB_roc_auc_ovr * 100))
    print("Precision : " + str(precision[0]))
    print("Recall : " + str(recall[0]))
    print("F1-Score : " + str(fscore[0]))
    print("Balanced Acc : " + str(bacc))
    print("Confusion Matrix : \n", cm)
    print("\n")

    total = [XGB_acc, XGB_roc_auc_ovo, XGB_roc_auc_ovr,
            precision[0], recall[0], fscore[0],
            bacc, cm]

    return total
