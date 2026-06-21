import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

try :
    from src import train, preprocessing
except ImportError :
    import train, preprocessing

import unittest

class TestModelInference(unittest.TestCase) :
    def setUp(self):
        
        print("\n\n")
        print("Test 단계 (test_pipeline.py)")
        print("\n\n")

        self.model = train.final_SVM
        self.X_test = preprocessing.X_test

        print("사용될 모델 : SVM\n")
        print("사용될 테스트 데이터 : X_test")
        print(self.X_test)


    def test_shape_is_equal(self) :
        print("\n")
        print("1. shape 확인 테스트를 진행합니다.")
        print("\n")

        y_pred = self.model.predict(self.X_test)
        print("y_pred 값")
        print(y_pred)

        self.assertEqual(y_pred.shape[0], self.X_test.shape[0])
        print("예측 결과의 shape와 입력 shape가 일치함을 확인되었습니다.\n")
        print("Test 상태 : 1번 Test 완료")

    def test_y_prob_range(self) :
        print("\n")
        print("2. 예측 확률 범위 테스트를 진행합니다.")
        print("\n")

        y_prob = self.model.predict_proba(self.X_test)
        print("y_prob 값")
        print(y_prob)

        for row in y_prob :
            for value in row :
                self.assertGreaterEqual(float(value), 0)
                self.assertLess(float(value), 1)
        print("예측 확률이 구간 [0, 1]에 있다는 것에 대하여 확인되었습니다.\n")
        print("Test 상태 : 2번 Test 완료")

    def test_input_values(self) :
        print("\n")
        print("3. 입력값 범위 검증 테스트를 진행합니다.")
        print("\n")


        print("3-1. 타겟 : 나이 (age)")
        for value in self.X_test['age'] :
            self.assertIn(value, range(0, 125))
        print("Test 상태 : 3-1번 Test 완료\n")


        print("3-2. 타겟 : 콜레스테롤 (chol)")
        for value in self.X_test['chol'] :
            self.assertIn(value, range(0, 600))
        print("Test 상태 : 3-2번 Test 완료\n")

        # # ?
        # # 일단 0 <= value && value < 5로 설정
        # print("3-3. 타겟 : 운동 후 휴식할 때 생기는 ST 분절 하강 수치 (oldpeak)")
        # for value in self.X_test['oldpeak'] :
        #     self.assertGreaterEqual(value, 0)
        #     self.assertLess(value, 5)
        # print("Test 상태 : 3-3번 Test 완료\n")

        print("Test 상태 : 3번 Test 완료")


if __name__ == '__main__' :
    unittest.main(verbosity=2)