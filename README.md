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
- Situation Labels 14

### 🚀 Models & Experiments
- Single-task models: CNN 기반으로 행동(Action), 감정(Emotion), 상황(Situation) 각각 독립 분류
- Multi-task model: 세 태스크를 통합 학습하여 학습 효율과 성능 개선 시도
- CNN-Temporal based multi-task video classification model : CNN과 Transformer 또는 LSTM을 결합한 시계열 기반 모델 추가 실험 진행 중

<details>
<summary>📝 Project Wrap-up(2025.06.24)</summary>

### 👍🏻 Good

- **모델 구조 다양화**: CNN-Based Single Task → CNN-Based Multi Task → CNN+LSTM Based Multi Task 구조로 확장하며 시계열 정보를 고려한 실험 진행.  
- **모듈화된 구조 설계**: 데이터 로딩, 모델 정의, 학습 루프 등을 함수 및 클래스로 모듈화하여 반복 실험이 용이하도록 설계함.  
- **실험 로깅 자동화**: `wandb`를 적극 활용하여 실험별 결과, 하이퍼파라미터, loss/accuracy track을 시각화하고 비교 분석 가능하게 함.  
- **Baseline 직접 구현 및 비교**: 학습용 CNN backbone을 직접 구성하거나 튜닝하며 baseline 성능과 비교 실험을 체계적으로 수행함.  

---

### 🙏🏻 Bad

- **데이터 전처리 및 분석 부족**: 모델 구조 설계에 비해 데이터 품질(프레임 내 고양이 미출현, 작은 크기, 결측 정보 등)에 대한 사전 분석 및 전처리가 부족했음.  
- **프레임 단위 라벨링 정확도 이슈**: 메타 데이터에 기반한 프레임 단위 라벨링이 실제 시각적 정보와 매칭되지 않아 학습 성능 저하로 이어짐.  
- **실험 시간 배분 실패**: 초기 Single Task 실험과 LLM 설계에 시간을 과도하게 사용하여 CNN-Temporal 모델 실험은 충분히 고도화하지 못함.  
- **작업 플로우 정교화 부족**: 전체 작업 흐름에 대한 트리거(예: 데이터 준비 완료 → 학습 시작 등)를 명확히 계획하지 않아 병렬적/효율적 작업이 어려웠음.  

---

### 👏🏻 Challenge & 개선할 점

- **멀티태스크 성능 향상**: 클래스 불균형 완화, 라벨 정제 등을 통해 멀티태스크 모델의 일반화 성능 개선 필요.  
- **데이터 품질 기반 필터링 도입**: 예) 고양이 객체 크기가 작거나 없을 경우 자동 필터링하여 학습에서 제외하는 방식 고려.  
- **시계열 라벨링 개선 방안 탐색**: 전체 프레임을 같은 라벨로 간주하는 단순 방식 대신, keypoint 기반 temporal segmentation 도입 검토 필요.  
- **플로우 관리 방식 개선**: 전 과정을 빠르게 1회 완성한 뒤, 그 위에 개선을 반복하는 방식으로 작업 플로우 전환.  

---

### ✍🏻 Keyword

- **Multi-Task Learning (Shared Encoding)**: 행동, 감정, 상황을 동시에 예측하는 방식으로 공유된 표현을 학습  
- **CNN + LSTM**: CNN으로 각 프레임 특징 추출 후 LSTM으로 시간 흐름 고려  
- **영상 기반 라벨링**: 프레임 단위 라벨링의 어려움과 시계열 구조 설계의 중요성 체감  
- **데이터 품질 이슈**: 학습 데이터의 대표성, 노이즈, 결측치가 학습 성능에 미치는 영향  

---

### ✅ To-Do

- **데이터 전처리 고도화 후 train/val/test 재구성**  
    - 동영상 단위로 프레임 수 통일  
    - 고양이 미출현, 객체 크기 작은 샘플 제거  
    - 클래스 불균형 완화 전략 수립  

- **테스크 재정의 및 라벨 정제**  
    - 모호한 라벨 또는 불확실한 샘플 제거  
    - 행동/감정/상황 간 경계가 명확한 샘플 위주로 정제  

- **개선된 데이터셋 기반으로 LSTM 모델 재학습**  
    - 시계열 흐름을 반영한 CNN+LSTM 구조 실험  
    - 멀티태스크에서 분기 헤드 구조 다양화 실험  

</details>


### 📁 Folder Structure
```
project_AI_PetBabel/      
├── task01_Classifier/          # 싱글 태스크 분류 
├── task02_MutiClassifier/      # 멀티 태스크 분류 
├── task03_CNN-Temporal/        # (추가작업) 시계열 비디오 프레임 분류
├── afterTask_Translator/       # (추후작업)반려동물 언어 번역기(LLM) 개발
└── sketch/                     # 작업 초안 및 base
```
