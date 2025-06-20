# Project : AI 🐾🐾
- 2025년 1학기 인공지능(AI) 수업 기말 프로젝트(초안)   
- 고양이의 행동 및 감정 상태를 분석하는 AI 모델 개발(멀티모달)   



### 📁 Data Structure
```
CAT_image/
├── 20201028_cat-arch-000156.mp4/                     
│   ├── 20201028_cat-arch-000156.mp4.json
│   ├── frame_0_timestamp_0.jpg
│   ├── frame_102_timestamp_6800.jpg
│   └──  ~.jpg
├── cat-arch-011926/   
│   ├── cat-arch-011926.json
│   ├── frame_10_timestamp_400.jpg
│   ├── frame_135_timestamp_5400.jpg
│   └──  ~.jpg                  
└── ~    

```

criterion_action = nn.CrossEntropyLoss()
criterion_emotion = nn.CrossEntropyLoss()
criterion_situation = nn.CrossEntropyLoss()

```
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

```