# Project : AI 🐾🐾
- Course : 2025년 1학기 인공지능(AI) 프로젝트  
- Subject : 고양이의 행동 및 감정 상태를 분석하는 AI 모델 개발(멀티모달)   
- Role : **Multi-Task Multi-Label** 모델 개발 및 실험

### 📁 Folder Structure
```
project_AI_PetBabel/      
├── task01_Classifier/          # 반려동물 분류 모델(이미지) - 싱글 모델
├── task02_MutiClassifier/      # 반려동물 분류 모델(이미지) 멀티 분류 
├── afterTask_Translator/       # 반려동물 언어 번역기(LLM) 개발
├── dataAnalysis.ipynb          # 데이터 전처리 및 분석(Multi)
└── requirements.txt    
```

### 🧪 Experimented Models
- 본 프로젝트에서는 다양한 CNN 기반 백본(backbone)을 사용하여 Multi-Task Multi-Label 분류 모델의 성능을 비교 실험함.   
- 각 모델은 공통 인코더를 기반으로 `action`, `emotion`, `situation`의 세 가지 라벨을 동시에 예측하도록 구성됨.

| Base Model         | Loss   | Macro F1 | Micro F1 | Partial Match Score | Exact Match Acc | Action Acc | Emotion Acc | Situation Acc |
|--------------------|--------|----------|----------|----------------------|------------------|-------------|--------------|----------------|
| **ResNet18**        | 0.3741 | 0.9650   | 0.9771   | 0.3310               | 0.9615           | 0.9743      | 0.9846       | 0.9725         |
| **ResNet50**        | 0.3012 | 0.9550   | 0.9722   | 0.3314               | 0.9497           | 0.9699      | 0.9820       | 0.9648         |
| **EfficientNet-B4** | 0.3799 | 0.9563   | 0.9697   | 0.3313               | 0.9431           | 0.9674      | 0.9780       | 0.9637         |
| **ConvNeXt-Base**   | 0.2139 | 0.9780   | 0.9847   | 0.3324               | 0.9725           | 0.9835      | 0.9912       | 0.9795         |

> 🔍 **결과 요약**:
- **ConvNeXt-Base**가 전반적인 성능(모든 지표)에서 가장 우수함을 보였음.
- ResNet18은 가장 가벼운 모델임에도 불구하고 비교적 높은 성능을 유지.
- 모든 모델에서 Exact Match Accuracy가 94% 이상으로, 세 가지 태스크를 동시에 예측하는 데에도 높은 정확도 달성.

