# Project : AI 🐾🐾
- 2025년 1학기 인공지능(AI) 수업 기말 프로젝트(초안)   
- 고양이의 행동 및 감정 상태를 분석하는 AI 모델 개발(멀티모달/이미지 분석)    


### 📦 Data Description
- AI-Hub : 반려동물 구분을 위한 동물 영상(https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=59)
- 데이터수 : 총 500만(강아지, 고양이)
    - emotion : 6개(공격성, 공포, 불안/슬픔, 편안, 행복/즐거움, 화남/불쾌)
    - behavior : 12개(아치, 스트레칭, 꾹꾹이 etc) 
        - **armstretch**데이터로 테스트
```
data/
├── CAT_labeled/               # armstretch 데이터셋
└── CAT_raw/
```


### 📁 Folder Structure
```
cat_behavior_project/
├── data/               # 데이터셋
├── dataset.py          # 데이터셋 클래스
├── model.py            # 모델 정의
├── train.py            # 학습 루프
├── eval.py             # 성능 평가
├── utils.py            # 
├── config.yaml         # 설정파일
└── requirements.txt
```

