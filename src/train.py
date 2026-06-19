# 전처리 했던 변수들
from preprocessing import X_train_scaled, y_train, X_test_scaled, y_test

# 앙상블로 특성 선택에 필요한 라이브러리
from sklearn.feature_selection import SelectFromModel

# 모델들
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# mlflow
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn

# 추론 후 점수들
from inference import LR_acc, LR_roc_auc_ovo, LR_roc_auc_ovr, \
    SVM_acc_list, SVM_roc_acc_ovo_list, SVM_roc_acc_ovr_list, \
    RF_acc, RF_roc_auc_ovo, RF_roc_auc_ovr, \
    KNN_acc, KNN_roc_auc_ovo, KNN_roc_auc_ovr, \
    XGB_acc, XGB_roc_auc_ovo, XGB_roc_auc_ovr

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

# =====================================
# 모델 학습 단계
# =====================================

# 실험 준비 (tracking 준비)
client = MlflowClient()
experiment_id = client.create_experiment("Heart Disease Prediction(CardioCare datasets used)")

experiment = client.get_experiment(experiment_id)

print("모델 실험 ID : ", experiment_id)

# 실험 이름 설정
mlflow.set_experiment(experiment_id=experiment_id)

# 실험 시작
with mlflow.start_run() :
    # Logistic Regression 학습
    LR_model = LogisticRegression(max_iter=300, random_state=1)
    LR_model.fit(X_train_selected, y_train)
    
    mlflow.log_params({"max_iter" : 300, "random_state" : 1})

    mlflow.sklearn.log_model(LR_model, "model")

    mlflow.log_metric("Accuracy Score", LR_acc)
    mlflow.log_metric("ROC AUC(OVO)", LR_roc_auc_ovo)
    mlflow.log_metric("ROC AUC(OVR)", LR_roc_auc_ovr)

    print("Logistic Regression 모델 학습 완료")

    print("\n")

    # SVM(Support Vector Machine) 학습
    SVM_kernel = ['linear', 'poly', 'rbf', 'sigmoid'] # precomputed 커널은 데이터가 정방행렬만 가능
    SVM_fitted_list = []

    for i in range(0, len(SVM_kernel)) :
        # SVC의 probality 속성이 scikit-learn 1.11 버전부터 deprecated 됨에 따라
        # CalibratedClassifierCV() 함수로 이를 대체함

        if (i == 'poly') :
            SVM_model = CalibratedClassifierCV(SVC(kernel=SVM_kernel[i], C=5, degree=3, random_state=1), ensemble=False)
        else :
            SVM_model = CalibratedClassifierCV(SVC(kernel=SVM_kernel[i], C=5, random_state=1), ensemble=False)

        SVM_model.fit(X_train_selected, y_train)

        # 학습된 모델을 리스트화 해서 그대로 inference로 넘김
        SVM_fitted_list.insert(i, SVM_model)

        if (i == 'poly') :
            mlflow.log_params({"kernel" : SVM_kernel[i],
                               "C" : 5,
                               "degree" : 3,
                               "random_state" : 1})
        else :
            mlflow.log_params({"kernel" : SVM_kernel[i],
                               "C" : 5,
                               "random_state" : 1})
        
        mlflow.sklearn.log_model(SVM_model, "model")

        mlflow.log_metric("Accuracy Score", SVM_acc_list[i])
        mlflow.log_metric("ROC AUC(OVO)", SVM_roc_acc_ovo_list[i])
        mlflow.log_metric("ROC AUC(OVR)", SVM_roc_acc_ovr_list[i])

    print("SVM(Support Vector Machine) 모델 학습 완료")

    print("\n")

    # Random Forest 학습 (앙상블 방법에서 사용된 모델 가져옴)
    RF_model = forest
    RF_model.fit(X_train_selected, y_train)

    mlflow.log_params({"n_estimators" : 100, "max_depth" : 5, "random_state" : 1})

    mlflow.sklearn.log_model(RF_model, "model")

    mlflow.log_metric("Accuracy Score", RF_acc)
    mlflow.log_metric("ROC AUC(OVO)", RF_roc_auc_ovo)
    mlflow.log_metric("ROC AUC(OVR)", RF_roc_auc_ovr)

    print("Random Forest 모델 학습 완료")
    print("\n")

    # KNN(K-Nearest Neighbors) 학습
    KNN_model = KNeighborsClassifier(n_neighbors=10)
    KNN_model.fit(X_train_selected, y_train)

    mlflow.log_param("n_neighbors", 10)

    mlflow.sklearn.log_model(KNN_model, "model")

    mlflow.log_metric("Accuracy Score", KNN_acc)
    mlflow.log_metric("ROC AUC(OVO)", KNN_roc_auc_ovo)
    mlflow.log_metric("ROC AUC(OVR)", KNN_roc_auc_ovr)

    print("KNN(K-Nearest Neighbors) 모델 학습 완료")
    print("\n")

    # XGBoost(eXtra Gradient Boost) 학습 (설정은 Random Forest와 똑같이 하였음)
    XGB_model = XGBClassifier(n_estimators=100, max_depth=5, random_state=1)
    XGB_model.fit(X_train_selected, y_train)

    mlflow.log_params({"n_estimators" : 100, "max_depth" : 5, "random_state" : 1})
    
    mlflow.sklearn.log_model(XGB_model, "model")

    mlflow.log_metric("Accuracy Score", XGB_acc)
    mlflow.log_metric("ROC AUC(OVO)", XGB_roc_auc_ovo)
    mlflow.log_metric("ROC AUC(OVR)", XGB_roc_auc_ovr)

    print("XGBoost (eXtra Gradient Boost) 분류기 모델 학습 완료")
    print("\n")
