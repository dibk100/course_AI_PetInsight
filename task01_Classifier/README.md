# Task01_Classifier 🐾🐾
반려동물(고양이) 분류 single task 모델 개발

### 🔄 TASK
- 행동(Action), 감정(Emotion), 상황(Situation) 분류

### 📦 Data Description
- [AI-Hub : 반려동물 구분을 위한 동물 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=59)

```
Json info :
├── species(종)               # DOG, CAT(2)
├── action(행동)              # DOG (13) / CAT (12)
├── emotion(감정)             # 행복/즐거움,편안/안정,불안/슬픔,화남/불쾌,공포,공격성 (6)
├── owner(반려인 정보)            
│   └── situation(촬영 상황)  # emotion(감정)과 페어됨.
│
└── inspect(관찰 객체)        
    ├── cemotion(감정)    
    └── action(행동)
```


### 📁 Folder Structure
```
task01_Classifier/      
├── data/               # 데이터셋
├── dataset.py          # 데이터 로딩 및 전처리
├── model.py            # 모델들 정리
├── train.py            # 학습 루프
├── eval.py             # 성능 평가
├── main.py             # 전체 파이프라인 실행 (학습 + 평가)
├── outputs/            # 모델 저장, 예측 결과 등 출력물 저장용
│   └── checkpoints/    # 학습된 모델 저장
└── requirements.txt    # 추후에 작성하기
```

