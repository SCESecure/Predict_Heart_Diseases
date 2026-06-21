try :
    from preprocessing import X_test, y_test
    from train import final_KNN

except ImportError :
    from src.preprocessing import X_test, y_test
    from src.train import final_KNN


import logging

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
    prediction = final_KNN.predict(instance)

    print("prediction")
    print(prediction)
    print("\n")


    print(f'Inst.{i} -> Model: 1.0, X_test.shape : {X_test.shape}, Pred: {prediction[0]}, Real: {y_test.iloc[i]}')
    logging.info(f'Inst.{i} -> Model: 1.0, X_test.shape : {X_test.shape}, Pred: {prediction[0]}, Real: {y_test.iloc[i]}')
