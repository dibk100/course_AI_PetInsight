# Project : AI 🐾🐾
- 2025년 1학기 인공지능(AI) 수업 기말 프로젝트(초안)   
- 고양이의 행동 및 감정 상태를 분석하는 AI 모델 개발(멀티모달)   

### 🔄 To-Do
- Task01 : 반려동물 분류 모델(이미지) 개발
- Task02 : 반려동물 언어 번역기(LLM) 개발
    - ISSUE : 언어 해석기 -> 번역기로 변경 시, 번역이 실제로 맞는지 성능 평가 어려움(전문가 필요)

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

### 📁 Data Structure
```
data/
├── action/                     
│   ├── ARCH/
│   │   ├── 20201028_cat-arch-000156.mp4/
│   │   │  ├── 20201028_cat-arch-000156.mp4.json
│   │   │  ├── frame_0_timestamp_0.jpg
│   │   │  └──  frame_12_timestamp_800.jpg
│   │   ├── 20201113_cat-arch-000157.mp4/
│   │   │  ├── 20201113_cat-arch-000157.mp4.json
│   │   │  ├── ~~.jpg
│   │   │  └──  ~~.jpg
│   ├── ARMSTRETCH/
│   │   ├── cat-armstretch-044713/
│   │   │  ├── cat-armstretch-044713.json
│   │   │  └── frame_90_timestamp_3000.jpg
│   │   ├── 20201113_cat-arch-000157.mp4/
│   │   │  ├── 20201113_cat-arch-000157.mp4.json
│   │   │  ├── ~~.jpg
│   │   │  └──  ~~.jpg
│   └── ~~~/
│          ├── ~~.jpg
│          └──  ~~.jpg
├── emotion/                     
│   ├── EMOTION_공격성/
│   │   ├── cat-armstretch-080706/
│   │   │  ├── frame_5_timestamp_200.jpg
│   │   │  ├── ~~~.jpg
│   │   │  └── frame_30_timestamp_1200.jpg
│   │   └── cat-armstretch-080997/
│   │      ├── ~~.jpg
│   │      └──  ~~.jpg
│   ├── EMOTION_공포/
│   │   ├── 20201113_cat-arch-000157.mp4/
│   │   │  ├── ~~.jpg
│   │   │  └──  ~~.jpg
│   │   └── cat-arch-013235/
│   │      ├── ~~.jpg
│   │      └──  ~~.jpg
│   └── ~~~/
│          ├── ~~.jpg
│          └──  ~~.jpg
├── situation/                     
│   ├── SITUATION_밥그릇,장난감과같은소유물을만질때/
│   │   └── cat-getdown-071013/
│   │      ├── ~~~.jpg
│   │      └── ~~.jpg
│   └── SITUATION_빗질_발톱깍기_목욕등위생관리를할때/
│       ├── cat-tailing-022462/
│       │  ├── ~~.jpg
│       │  └──  ~~.jpg
│       └── cat-tailing-023607/
│          ├── ~~.jpg
│          └──  ~~.jpg
└──     

```