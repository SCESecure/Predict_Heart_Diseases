# 파일 불러오기
try :
    from preprocessing import X_train, X_train_scaled, y_train, X_test_scaled, y_test
    from inference import LR_predict, SVM_predict, RF_predict, KNN_predict, XGB_predict

except ImportError :
    from src.preprocessing import X_train, X_train_scaled, y_train, X_test_scaled, y_test
    from src.inference import LR_predict, SVM_predict, RF_predict, KNN_predict, XGB_predict

# 앙상블로 특성 선택에 필요한 라이브러리
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBClassifier
# cuda 사용할 경우 아래 코드 사용
# model = XGBClassifier(
#     device='cuda',        # GPU 사용
#     tree_method='hist',   # 2.0+ 에서는 hist + device 조합
#     n_estimators=300,
# )

# 모델들
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# mlflow
import mlflow
import mlflow.sklearn

# 추론 후 점수들
# from inference import LR_predict, \
#     SVM_acc_list, SVM_roc_acc_ovo_list, SVM_roc_acc_ovr_list, \
#     RF_acc, RF_roc_auc_ovo, RF_roc_auc_ovr, \
#     KNN_acc, KNN_roc_auc_ovo, KNN_roc_auc_ovr, \
#     XGB_acc, XGB_roc_auc_ovo, XGB_roc_auc_ovr

# 교차 검증
from sklearn.model_selection import KFold, GridSearchCV

# 그 이외의 라이브러리
import numpy as np

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

# 실험 이름 설정

mlflow.set_experiment("Heart Disease Prediction(CardioCare datasets used) [Logistic Regression]")
mlflow.autolog()

# LR 실험 시작
with mlflow.start_run() :
    # Logistic Regression 학습
    LR_model = LogisticRegression(max_iter=300, random_state=1)
    LR_model.fit(X_train_selected, y_train)
    print("Logistic Regression 모델 학습 완료")
    print("\n")

    mlflow.log_params({"max_iter" : 300, "random_state" : 1})

    mlflow.log_metric("Accuracy_Score", LR_predict(LR_model, X_test_selected, y_test)[0])
    mlflow.log_metric("ROC_AUC_OVO", LR_predict(LR_model, X_test_selected, y_test)[1])
    mlflow.log_metric("ROC_AUC_OVR", LR_predict(LR_model, X_test_selected, y_test)[2])

    mlflow.sklearn.log_model(LR_model, "model")

    print("Logistic Regression 모델 실험 완료\n")


mlflow.set_experiment("Heart Disease Prediction(CardioCare datasets used) [SVM]")
mlflow.autolog()

with mlflow.start_run() :
    # SVM(Support Vector Machine) 학습
    # SVM_kernel = ['linear', 'poly', 'rbf', 'sigmoid'] # precomputed 커널은 데이터가 정방행렬만 가능
    # SVM_fitted_list = []

    SVM_model = CalibratedClassifierCV(SVC(kernel='rbf', C=5, random_state=1), ensemble=False)
    SVM_model.fit(X_train_selected, y_train)

    print("SVM(Support Vector Machine) 모델 학습 완료")
    print("\n")


    mlflow.log_params({"kernel" : 'rbf',
                        "C" : 5,
                        "random_state" : 1})
        
    mlflow.sklearn.log_model(SVM_model, "model", skops_trusted_types=[
        "sklearn.calibration._CalibratedClassifier",
        "sklearn.calibration._SigmoidCalibration",
    ])

    mlflow.log_metric("Accuracy_Score", SVM_predict(SVM_model, X_test_selected, y_test)[0])
    mlflow.log_metric("ROC_AUC_OVO", SVM_predict(SVM_model, X_test_selected, y_test)[1])
    mlflow.log_metric("ROC_AUC_OVR", SVM_predict(SVM_model, X_test_selected, y_test)[2])

    print("SVM(Support Vector Machine) 모델 실험 완료\n")


mlflow.set_experiment("Heart Disease Prediction(CardioCare datasets used) [Random Forest]")
mlflow.autolog()

with mlflow.start_run() :
    # Random Forest 학습 (앙상블 방법에서 사용된 모델 가져옴)
    RF_model = forest
    RF_model.fit(X_train_selected, y_train)

    print("Random Forest 모델 학습 완료")
    print("\n")

    mlflow.log_params({"n_estimators" : 100, "max_depth" : 5, "random_state" : 1})

    mlflow.sklearn.log_model(RF_model, "model")

    mlflow.log_metric("Accuracy_Score", RF_predict(RF_model, X_test_selected, y_test)[0])
    mlflow.log_metric("ROC_AUC_OVO", RF_predict(RF_model, X_test_selected, y_test)[1])
    mlflow.log_metric("ROC_AUC_OVR", RF_predict(RF_model, X_test_selected, y_test)[2])

    print("Random Forest 모델 실험 완료\n")

mlflow.set_experiment("Heart Disease Prediction(CardioCare datasets used) [KNN]")
mlflow.autolog()

with mlflow.start_run() :
    # KNN(K-Nearest Neighbors) 학습
    KNN_model = KNeighborsClassifier(n_neighbors=10)
    KNN_model.fit(X_train_selected, y_train)

    print("KNN(K-Nearest Neighbors) 모델 학습 완료")
    print("\n")

    mlflow.log_param("n_neighbors", 10)

    mlflow.sklearn.log_model(KNN_model, "model", skops_trusted_types=[
        'sklearn.metrics._dist_metrics.EuclideanDistance64', 
        'sklearn.neighbors._kd_tree.KDTree'
    ])

    mlflow.log_metric("Accuracy_Score", KNN_predict(KNN_model, X_test_selected, y_test)[0])
    mlflow.log_metric("ROC_AUC_OVO", KNN_predict(KNN_model, X_test_selected, y_test)[1])
    mlflow.log_metric("ROC_AUC_OVR", KNN_predict(KNN_model, X_test_selected, y_test)[2])

    print("KNN(K-Nearest Neighbors) 모델 실험 완료\n")


mlflow.set_experiment("Heart Disease Prediction(CardioCare datasets used) [XGBoost]")
mlflow.autolog()

with mlflow.start_run() :
    # XGBoost(eXtra Gradient Boost) 학습 (설정은 Random Forest와 똑같이 하였음)
    XGB_model = XGBClassifier(n_estimators=100, max_depth=5, random_state=1)
    XGB_model.fit(X_train_selected, y_train)

    print("XGBoost (eXtra Gradient Boost) 분류기 모델 학습 완료")
    print("\n")

    mlflow.log_params({"n_estimators" : 100, "max_depth" : 5, "random_state" : 1})
    
    mlflow.sklearn.log_model(XGB_model, "model", skops_trusted_types=[
        'xgboost.core.Booster', 
        'xgboost.sklearn.XGBClassifier'
    ])

    mlflow.log_metric("Accuracy_Score", XGB_predict(XGB_model, X_test_selected, y_test)[0])
    mlflow.log_metric("ROC_AUC_OVO", XGB_predict(XGB_model, X_test_selected, y_test)[1])
    mlflow.log_metric("ROC_AUC_OVR", XGB_predict(XGB_model, X_test_selected, y_test)[2])

    print("XGBoost (eXtra Gradient Boost) 분류기 모델 실험 완료\n")


print("\n")
print("교차 검증 및 하이퍼파라미터 튜닝합니다.")
print("\n")

def grid_searching(grid_target_model, param_grid, kf) :

    # CPU를 기준으로 돌아갑니다.
    cv = GridSearchCV(grid_target_model, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=2)

    cv.fit(X_train_selected, y_train)

    result = {"best_params" : cv.best_params_,
              "best_score" : cv.best_score_}

    return result

# 교차 검증
kf = KFold(n_splits=5, shuffle=True, random_state=1)
grid_target_model_list = {"LR" : LogisticRegression(),
                          "SVM" : SVC(),
                          "RF" : RandomForestClassifier(),
                          "KNN" : KNeighborsClassifier(),
                          "XGB" : XGBClassifier()}

# print('\n')
# print("대상 : Logistic Regression 모델")
# print('\n')

# mlflow.set_experiment("Heart Disease Prediction(CardioCare datasets used) [LR-5Fold]")
# mlflow.autolog()

# with mlflow.start_run() :

#     param_grid = {"penalty" : ['l1', 'l2', 'elasticnet', None],
#               "C" : np.arange(1, 60, 3),
#               "l1_ratio" : np.arange(0.0001, 1, 10),
#               "dual" : [True, False],
#               "fit_intercept" : [True, False],
#               "solver" : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
#               "max_iter" : np.arange(100, 1000, 10)
#               }

#     cv = grid_searching(grid_target_model_list["LR"], param_grid, kf)

#     mlflow.log_params(cv['best_params'])

#     mlflow.log_metric("CV_best_score", cv["best_score"])

#     print("Logistic Regression 모델의 최적의 하이퍼파라미터는 다음과 같습니다.\n", cv["best_params"])
#     print("점수는 다음과 같습니다.\n", cv["best_score"])

# print('\n')
# print("대상 : SVM 모델")
# print('\n')
# mlflow.set_experiment("Heart Disease Prediction(CardioCare datasets used) [SVM-5Fold]")
# mlflow.autolog()

# with mlflow.start_run() :

#     param_grid = {"C" : np.arange(1, 60, 3),
#                   "kernel" : ['linear', 'poly', 'rbf', 'sigmoid'],
#                   "degree" : np.arange(1, 20, 2), # 어차피 poly가 아닌 다른 parameter들은 무시됨
#                   "gamma" : ['scale', 'auto'],
#                   "verbose" : [True, False],
#                   "max_iter" : np.arange(1, 100, 2),
#               }

#     cv = grid_searching(grid_target_model_list["SVM"], param_grid, kf)

#     mlflow.log_params(cv['best_params'])

#     mlflow.sklearn.log_model(grid_target_model_list["SVM"], "model", skops_trusted_types=[
#         "sklearn.calibration._CalibratedClassifier",
#         "sklearn.calibration._SigmoidCalibration",
#     ])

#     mlflow.log_metric("CV_best_score", cv["best_score"])

#     print("SVM 모델의 최적의 하이퍼파라미터는 다음과 같습니다.\n", cv["best_params"])
#     print("점수는 다음과 같습니다.\n", cv["best_score"])

print("\n")
print("대상 : RF 모델")
print("\n")

mlflow.set_experiment("Heart Disease Prediction(CardioCare datasets used) [RF-5Fold]")
mlflow.autolog(log_models=False) # 이렇게 안하면 메모리가 죽음

with mlflow.start_run() :

    # 9 * 10 * 3 = 270
    param_grid = {"n_estimators" : np.arange(100, 500, 50),
                  "max_depth" : np.arange(5, 50, 5),
                  "max_features" : ["sqrt", "log2", None],
                  }
    cv = grid_searching(grid_target_model_list["RF"], param_grid, kf)

    mlflow.log_params(cv['best_params'])

    mlflow.log_metric("CV_best_score", cv["best_score"])

    rf_cv_best_params = cv['best_params']
    rf_cv_best_score = cv["best_score"]

    print("Random Forest 모델의 최적의 하이퍼파라미터는 다음과 같습니다.\n", cv["best_params"])
    print("점수는 다음과 같습니다.\n", cv["best_score"])


print("\n")
print("대상 : KNN 모델")
print("\n")

mlflow.set_experiment("Heart Disease Prediction(CardioCare datasets used) [KNN-5Fold]")
mlflow.autolog(log_models=False)

with mlflow.start_run() :

    # 10 * 27 * 2 = 540
    param_grid = {"n_neighbors" : np.arange(5, 50, 5),
                  "leaf_size" : np.arange(30, 300, 10),
                  "p" : [1, 2],
                  }
    cv = grid_searching(grid_target_model_list["KNN"], param_grid, kf)

    mlflow.log_params(cv['best_params'])

    mlflow.sklearn.log_model(grid_target_model_list["KNN"], "model", skops_trusted_types=[
        'sklearn.metrics._dist_metrics.EuclideanDistance64', 
        'sklearn.neighbors._kd_tree.KDTree'
    ])

    knn_cv_best_params = cv['best_params']
    knn_cv_best_score = cv["best_score"]

    mlflow.log_metric("CV_best_score", cv["best_score"])

    print("KNN 모델의 최적의 하이퍼파라미터는 다음과 같습니다.\n", cv["best_params"])
    print("점수는 다음과 같습니다.\n", cv["best_score"])


# print("\n")
# print("대상 : XGB 모델")
# print("\n")

# mlflow.set_experiment("Heart Disease Prediction(CardioCare datasets used) [XGB-5Fold]")
# mlflow.autolog()

# with mlflow.start_run() :

#     param_grid = {"booster" : ['gbtree'],
#                   "max_depth" : np.arange(1, 10),
#                   "min_child_weight" : np.arange(1, 10),
#                   "gamma": np.arange(1, 10),
#                   "n_estimators" : np.arange(10, 200, 5)
#                   }
#     cv = grid_searching(grid_target_model_list["XGB"], param_grid, kf)

#     cv.fit(X_train_selected, y_train)

#     mlflow.log_params(cv['best_params'])

#     mlflow.sklearn.log_model(grid_target_model_list["XGB"], "model", skops_trusted_types=[
#         'xgboost.core.Booster', 
#         'xgboost.sklearn.XGBClassifier'
#     ])

#     mlflow.log_metric("CV_best_score", cv["best_score"])

#     print("XGBoost 모델의 최적의 하이퍼파라미터는 다음과 같습니다.\n", cv["best_params"])
#     print("점수는 다음과 같습니다.\n", cv["best_score"])

print("\n")
print("하이퍼파라미터 튜닝 완료")
print("최종 결과 입니다. (소괄호 안 값은 점수입니다.))\n")
print("Random Forest 모델 : ", rf_cv_best_params, " (", rf_cv_best_score,")\n")
print("KNN 모델 : ", knn_cv_best_params, " (", knn_cv_best_score,")\n")
print("\n")

final_RF = RandomForestClassifier(max_depth = int(rf_cv_best_params['max_depth']), 
                                  max_features = rf_cv_best_params['max_features'], 
                                  n_estimators = int(rf_cv_best_params['max_depth'])
                                  ).fit(X_train, y_train)
final_KNN = KNeighborsClassifier(leaf_size = int(knn_cv_best_params['leaf_size']), 
                                 n_neighbors = int(knn_cv_best_params['n_neighbors']), 
                                 p = knn_cv_best_params['p']).fit(X_train, y_train)
