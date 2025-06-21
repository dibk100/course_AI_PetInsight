# project_AI_PetBabel 🐾🐾
- Course : 2025년 1학기 인공지능(AI) 프로젝트  
- Subject :A Multi-task AI Model for Understanding Cat Behavior and Emotion(고양이 행동과 감정 이해를 위한 다중 태스크 AI 모델 개발 )

- Role : 
   - 단익 : **Multi-Task Multi-Label** 모델 개발 및 실험, 추가 실험
   - 유라 : **Single-Task(emotion/situation) Multi-Label** 모델 개발 및 실험
   - 태윤 : **Single-Task(action) Multi-Label** 모델 개발 및 실험
   - 현준 : **Data analysis and processing** 데이터 총괄

### ⚙️ Tasks
- 행동(Action), 감정(Emotion), 상황(Situation) 분류

### 📦 Data Description
- [AI-Hub : 반려동물 구분을 위한 동물 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=59)
- Action Labels 12
- Emotion Labels 6
- Situation Labels 17

### 🚀 Models & Experiments
- Single-task models: CNN 기반으로 행동(Action), 감정(Emotion), 상황(Situation) 각각 독립 분류
- Multi-task model: 세 태스크를 통합 학습하여 학습 효율과 성능 개선 시도
- CNN-Temporal based multi-task video classification model : CNN과 Transformer 또는 LSTM을 결합한 시계열 기반 모델 추가 실험 진행 중

### 📁 Folder Structure
```
project_AI_PetBabel/      
├── task01_Classifier/          # 싱글 태스크 분류 
├── task02_MutiClassifier/      # 멀티 태스크 분류 
├── task03_CNN-Temporal/        # (추가작업) 시계열 비디오 프레임 분류
├── afterTask_Translator/       # (추후작업)반려동물 언어 번역기(LLM) 개발
└── sketch/                     # 작업 초안 및 base
```


