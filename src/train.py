from preprocessing import X_train_scaled, X_test_scaled, y_train, y_test, data_col

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import pandas as pd

# RandomForest 사용
forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
forest.fit(X_train_scaled, y_train)

# 특성 선택
selector = SelectFromModel(forest, prefit=True)
features_bool = selector.get_support()

selected_features = []
for i in range(0,X_train_scaled.shape[1]) :
    if (features_bool[i] == True) :
        selected_features.insert(i, X_train_scaled.columns[i])

# 선택된 특성 출력 (selected_features)
print("RandomForest에 의해 선택된 특성은 다음과 같습니다 : " + selected_features)

