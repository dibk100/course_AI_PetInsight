# Task01_Classifier 🐾🐾
반려동물(고양이) 분류 single task 모델 개발

### 🔄 TASK
- 행동(Action), 감정(Emotion), 상황(Situation) 분류

### 📦 Data Description
- [AI-Hub : 반려동물 구분을 위한 동물 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=59)

```
Json info :
├── species(종)               # CAT
├── action(행동)              # 12가지 행동 라벨
├── emotion(감정)             # 6가지 감정 라벨
├── owner(반려인 정보)            
│   └── situation(촬영 상황)  # inspect의 emotion(감정)과 페어함.
│
└── inspect(관찰 객체)        
    ├── emotion(감정)    
    └── action(행동)
```

### 📁 Folder Structure
```
task01_Classifier/      
├── Action_preprocess_classification_ResNet18.ipynb   
├── Action_classification_EfficientNet-b0.ipynb
├── Action_classification_ResNet50.ipynb
├── Action_classification_ViTB16.ipynb
├── Action_inference_EfficientNet-b0.ipynb
└── Emotion_Situation_classificaton.ipynb             
```

