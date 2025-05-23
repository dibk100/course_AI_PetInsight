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
├── CAT_labeled/
│   └── *.json                ← 각 동영상의 라벨링 정보가 담긴 JSON 파일
├── CAT_raw/
│   └── [동영상이름]/         ← 각 동영상별 프레임 이미지가 들어있는 폴더
│       └── frame_*.jpg      ← 프레임 번호 및 타임스탬프가 붙은 이미지
```


### 📁 Folder Structure
```
cat_behavior_project/
├── data/               # 데이터셋
├── dataset.py          # 데이터 로딩 및 전처리
├── model.py            # 모델들 정리
├── train.py            # 학습 루프
├── eval.py             # 성능 평가
├── main.py             # 전체 파이프라인 실행 (학습 + 평가)
│
├── outputs/            # 모델 저장, 예측 결과 등 출력물 저장용
│   ├── checkpoints/    # 학습된 모델 저장
│   └── logs/           # 로그 파일 저장
└── requirements.txt    # 추후에 작성하기
```

