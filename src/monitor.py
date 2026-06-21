try :
    from preprocessing import X_test, y_test
    from train import final_SVM

except ImportError :
    from src.preprocessing import X_test, y_test
    from src.train import final_SVM

from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score

import logging

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    filename='cardiocare.log', 
    level=logging.INFO
    )

print("\n\n")
print("Monitor 단계 (monitor.py)")
print("\n\n")

print("X_test")
print(X_test)
print("X_test.shape[0] : ", X_test.shape[0])
print("\n")

for i in range(X_test.shape[0]) :
    print(f"X_test.iloc[{i}, :] : ")
    print(X_test.iloc[i, :])
    print('\n')
    instance = X_test.iloc[i, :].values.reshape(1, -1)
    prediction = final_SVM.predict(instance)

    print("prediction")
    print(prediction)
    print("\n")


    print(f'Inst.{i} -> Model: 1.0, X_test.shape : {X_test.shape}, Pred: {prediction[0]}, Real: {y_test.iloc[i]}')
    logging.info(f'Inst.{i} -> Model: 1.0, X_test.shape : {X_test.shape}, Pred: {prediction[0]}, Real: {y_test.iloc[i]}')

print("\n")
print("X_test 복사본(이하 X_test_cpy)에서 특성 하나를 shift")
print("방식 : chol 특성의 평균 이동")
print("\n")

# chol 특성 평균을 +30

X_test_cpy = X_test.copy()
sigma = X_test['chol'].std()

X_test_cpy['chol'] = X_test['chol'] + sigma * 30

print("변경된 X_test(X_test_cpy)")
print(X_test_cpy)

print("\n완료")

print("\n")
print("Kolmogorov-Smirnov 검증 시작합니다.")
print("\n")

def KS_testing(sample1, sample2) :
    test_statistic, p_value = ks_2samp(sample1, sample2)

    print("test_statistic : ", test_statistic)
    print("p_value : ", p_value)

    drift_list = []
    drift_cnt = 0

    for i in range(p_value.shape[0]) :
        if p_value[i] < 0.05 :
            drift_list.insert(i, sample1.columns[i])
            drift_cnt += 1
            print(f"{sample1.columns[i]} 특성에 대하여 데이터 드리프트 의심됨!\n")

    if drift_cnt > 0 :
        print(f"결론 : {drift_cnt}개의 특성에 데이터 드리프트가 의심됩니다.\n")
        print("의심되는 특성은 다음과 같습니다.")
        print(drift_list)

    else :
        print("결론 : 모든 특성에 대한 데이터가 동일 분포로 추정됩니다.\n")

    result = {"test_statistic" : test_statistic,
              "p_value" : p_value}
    return result


print("[X_test vs. X_test_cpy]")

KS_testing(X_test, X_test_cpy)

print("\n")
print("두 데이터 셋을 비교합니다.")
print("\n")

print("[balanced_accuracy]")

X_test_pred = final_SVM.predict(X_test)
X_test_cpy_pred = final_SVM.predict(X_test_cpy)

X_test_bacc = balanced_accuracy_score(y_test, X_test_pred)
X_test_cpy_bacc = balanced_accuracy_score(y_test, X_test_cpy_pred)

print("X_test : ", X_test_bacc)
print("X_test_cpy : ", X_test_cpy_bacc)

print("[가시화]")

idx = X_test['chol'].index
width = 0.25

x = np.arange(len(X_test))

# plt.bar(idx, X_test['chol'], width=width, color='red', label="X_test['chol']")
# plt.bar(idx + 0.25, X_test_cpy['chol'], width=width, color='green', label="X_test_cpy['chol']")

plt.scatter(x, X_test['chol'], color='red', label="X_test['chol']")
plt.scatter(x, X_test_cpy['chol'], color='green', label="X_test['chol']")
plt.legend()

plt.xlabel('tester')
plt.ylabel('chol')


plt.savefig("./X_test-X_test_cpy.png")

print("가시화 완료")

