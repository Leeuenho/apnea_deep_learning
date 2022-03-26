# apnea_deep_learning


심박수를 통해 수면 무호흡을 예상하고, 수면 무호흡 심각성을 확인하여 치료 권장 여부를 확인하는 머신러닝 구현


학습 및 검증 데이터 : https://physionet.org/content/apnea-ecg/1.0.0/
테스트 데이터 : 실제 병원에서 확보한 수면 다원검사결과 이용



#
# 머신 검증
![x_data_confusion_metrix](https://user-images.githubusercontent.com/80818827/160247618-cb212684-d254-4e9b-a016-2ece185393fe.png)

#
# 실제 수면 다원검사 결과 이용한 검증
![real_data_confusion_metrix](https://user-images.githubusercontent.com/80818827/160247625-9512a09f-42ea-4827-9b4f-2487b4e681b6.png)

#
# 한명의 환자 전체 수면 시간 중 실제 apnea와 예측 apnea

사용한 data = x03

AHI = 0.13/ 예측 값 = 27 

노란색 = 실제 apnea이고, model이 apnea라고 예측 (실제 apnea 구간 맞춤)

붉은색 = 실제 apnea 이지만 model이 non-apnea라고 예측 (실제 anpea지만 틀림)

파란색 = 실제 non-apena이지만 model이 apena라고 예측

 
![hrv+apnea](https://user-images.githubusercontent.com/80818827/160247713-350cbdc1-b17e-465b-907d-3b7cce7b4411.png)
