한국어 | [English](README.en.md)

# EEG2SDiff — 뇌파 표현학습 기반 신경 이미지 디코딩

뇌파에서 사람이 보거나 떠올린 이미지를 복원하는 것이 학부 졸업연구 주제였습니다. 이 저장소는
그 파이프라인의 앞단, EEG 신호를 이미지 클래스가 구분되는 잠재 표현으로 바꾸는 인코더 부분입니다.

발표: 한국지능정보시스템학회 2023 춘계학술대회 구두발표,
"인지 시뮬레이션을 위한 잠재 확산 기반 신경 디코딩 시스템".

## 이 저장소가 다루는 범위

전체 파이프라인은 EEG 인코더와 잠재 확산 이미지 디코더 두 단계입니다. 여기 올린 것은 앞 단계인
인코더뿐이고, 뒤쪽 디코더는 졸업연구 2단계라 이 저장소에는 없습니다. 완성된 도구라기보다 연구
과정에서 나온 코드로 봐주시면 됩니다.

## 동작

ThoughtViz EEG 데이터(14채널, 객체 10클래스)를 입력으로 받습니다. 모델 TripleNet은 2층
LSTM(32→128) 뒤에 Dense 사영과 L2 정규화를 붙여 임베딩을 뽑습니다. 학습은 semi-hard
triplet loss로, 같은 클래스 뇌파는 가깝게 다른 클래스는 멀게 공간을 잡습니다. 평가는 임베딩에
KMeans를 돌려 군집 정확도(Hungarian 매칭)를 재고, t-SNE로 클래스가 갈라지는 모습을 봅니다.

분류기를 바로 붙이지 않고 메트릭 러닝을 쓴 이유는 EEG가 잡음과 피험자 편차가 심하기 때문입니다.
클래스 구조가 거리로 드러나는 임베딩이, 뒤에 올 생성 디코더가 조건으로 쓰기에 더 낫습니다.

## 실행

```bash
python lstm_kmean/train.py        # TripleNet 학습 (TF/Keras + tensorflow-addons)
python lstm_kmean/inference.py    # 테스트셋 임베딩, KMeans 군집정확도, t-SNE
```

스택: TensorFlow/Keras, tensorflow-addons(TripletSemiHardLoss), scikit-learn.
학부 연구(2023)이며, 코드는 당시 상태 그대로 두었습니다.
