import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src import train

import unittest

class TestModelInference(unittest.TestCase) :
    def setUp(self):
        self.model = train.final_KNN
        self.X_test = train.X_test_selected
    
    def test_prediction_output_shape(self) :
        y_pred = self.model.predict(self.X_test)
        self.assertEqual(y_pred.shape[0], self.X_test.shape[0])
        print("Test 상태 : 예측 결과의 shape와 입력 shape가 일치함을 확인되었습니다.\n")
    

if __name__ == '__main__' :
    unittest.main()