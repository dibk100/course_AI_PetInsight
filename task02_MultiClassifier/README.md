# Task02_MultiClassifier
반려동물(고양이) Multi-Task_Multi-Labels Classification 모델 개발

## 📂 Dataset Info
- 총 JSON 파일 수(= 동영상 수): 829
- 각 태스크별 라벨 구성:
  - [ACTION] : 13개
  - [EMOTION] : 6개
  - [SITUATION] : 17개

## ✨ Data Preprocessing
- 라벨 조합별 빈도 기준:
  - frequency > 5: 해당 동영상에서 이미지 1장 무작위 추출
  - frequency ≤ 5: 해당 동영상의 모든 이미지 사용
- 최종 이미지 수: **18,172장**
    | 구분  | 개수 |
    |-------|------|
    | Train | 13,129 |
    | Val   | 2,317 |
    | Test  | 2,726 |

## 🧠 Model Architecture
<details> 
<summary>❌ TEST01: 단일 출력 (Multi-Output Classification)</summary>   

- **하나의 출력층**으로 모든 태스크 라벨 예측  
- 각 태스크별 로짓을 구분하지 않고 **공통 FC layer**로 3개 태스크를 한 번에 예측  
- 라벨 간의 상호 의존성을 반영하기 어려움
      
      [입력 이미지]
            ↓
      [공통 인코더: CNN backbone (예: ResNet)]
            ↓
      [단일 Fully Connected Layer → 총 36 로짓 출력]
            ↓
      ──────────────────────────────
      | action logits (13)          |
      | emotion logits (6)          |
      | situation logits (17)       |
      ──────────────────────────────
      
</details> 

### ✅ TEST02: 멀티 출력 + 공통 인코딩 구조 (Multi-Head)
- 공통 인코더 + 분리된 출력 헤드
- 특징 벡터는 공유하되, 라벨별로 개별 FC layer 구성
- 라벨 간 독립성과 상호 관계를 모두 반영할 수 있는 구조    
- 즉, 하나의 이미지 인코딩에서 여러 개의 태스크를 동시에 처리하는 구조
```python
[입력 이미지]
        ↓
[shared encoder: CNN backbone (예: ResNet)]
        ↓ (공통 특성 추출된 feature vector, 예: (B, 512) or (B, 2048))
    ┌────────────┬──────────────┬───────────────┐
↓            ↓              ↓
action_head  emotion_head   situation_head  ← (각각 FC layer)
↓            ↓              ↓
13 logits    6 logits       17 logits       ← (raw output, 로짓)

```
        
모델 구조 요약 :
- `shared_encoder`: 이미지 특성 추출
- `action_head`, `emotion_head`, `situation_head`: 태스크별 분류기 (FC Layer)
- `forward` : 세 개 태스크 각각에 대해 로짓 반환 (`B x num_classes`)
- 손실함수 : nn.CrossEntropyLoss()
   

## 🚀 실행 예시
```
python inference.py --input_dir ./inference_data/test_images --config ./configs/base.yaml
```

## 🧪 실험 기록
| Metric                    | ResNet18 | ResNet50 | EfficientNet-B4 | ConvNeXt-Base |
|---------------------------|----------|----------|------------------|---------------|
| **Loss**                  | 0.3741   | 0.3012   | 0.3799           | **0.2139**    |
| **Macro F1**              | 0.9650   | 0.9550   | 0.9563           | **0.9780**    |
| **Micro F1**              | 0.9771   | 0.9722   | 0.9697           | **0.9847**    |
| **Partial Match Score**   | 0.3310   | 0.3314   | 0.3313           | **0.3324**    |
| **Exact Match Accuracy**  | 0.9615   | 0.9497   | 0.9431           | **0.9725**    |
| **Acc (Action)**          | 0.9743   | 0.9699   | 0.9674           | **0.9835**    |
| **Acc (Emotion)**         | 0.9846   | 0.9820   | 0.9780           | **0.9912**    |
| **Acc (Situation)**       | 0.9725   | 0.9648   | 0.9637           | **0.9795**    |
<details> 
<summary>Training Settings</summary>   

```python
batch_size: 16       
epochs: 30
learning_rate: 5e-5       
seed: 42
device: "cuda" 
```

</details> 
<details> 
<summary>Evaluation Metrics</summary>   

- macro_f1를 기준으로 best모델 저장함.

| 지표명                        | 설명                                                                 | 특징 |
|-----------------------------|----------------------------------------------------------------------|------|
| **Loss**                    | 모델 학습 중 최소화하는 손실 함수 값                                  | 낮을수록 예측과 실제 라벨 차이가 적음 |
| **Macro F1 Score**          | 클래스별 F1-score를 각각 계산한 후 단순 평균                           | 클래스 불균형에 덜 민감, 모든 클래스를 균등하게 평가 |
| **Micro F1 Score**          | 모든 클래스의 TP, FP, FN을 합산한 뒤 계산한 F1-score                   | 데이터셋 전체 성능 반영, 빈도 높은 클래스의 영향 큼 |
| **Partial Match Score**     | 멀티 태스크에서 일부 라벨만 맞춘 경우에 점수를 부여하는 지표            | 모든 태스크가 아닌 일부 일치에도 점수 부여 |
| **Exact Match Accuracy**    | 멀티 태스크에서 모든 라벨이 정확히 일치한 경우만 정답으로 판단         | 가장 엄격한 지표, 모든 태스크를 동시에 맞춰야 1점 |
| **Label-wise Accuracy**     | 각 태스크별(예: action, emotion, situation) 정확도                     | 태스크별 성능을 세부적으로 분석 가능 |

</details> 