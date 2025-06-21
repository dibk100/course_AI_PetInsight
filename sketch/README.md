# Project : AI 🐾🐾
- 2025년 1학기 인공지능(AI) 수업 기말 프로젝트(초안)   
- 고양이의 행동 및 감정 상태를 분석하는 AI 모델 개발(멀티모달)   

### 🔄 To-Do
- Task01 : 반려동물 분류 모델(이미지) 개발
- Task02 : 반려동물 언어 번역기(LLM) 개발
    - ISSUE : 언어 해석기 -> 번역기로 변경 시, 번역이 실제로 맞는지 성능 평가 어려움(전문가 필요)


### 📦 Data Description
- [AI-Hub : 반려동물 구분을 위한 동물 영상](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=59)

### 📁 Folder Structure
```
project_AI_PetBabel/      
├── TASK01_Classifier/               
├── TASK02_Translator/               
├── main.py                     # CLI 실행 파일
├── video_util.py               # [1] 입력 처리
├── vision_model.py             # [2] 감정/행동/상황 분류
├── llm_generator.py            # [3] 자연어 생성
└── requirements.txt    
```


```
Json info :
├── species(종)               # CAT
├── action(행동)              # CAT (12)
├── emotion(감정)             # 행복/즐거움,편안/안정,불안/슬픔,화남/불쾌,공포,공격성 (6)
├── owner(반려인 정보)            
│   └── situation(촬영 상황)  # emotion(감정)과 페어함.
│
└── inspect(관찰 객체)        
    ├── cemotion(감정)    
    └── action(행동)
```