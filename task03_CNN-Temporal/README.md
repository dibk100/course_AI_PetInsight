# task03_CNN-Temporal 🐾🐾
CNN-Temporal based multi-task video classification model   
🔄 TASK : 시계열 정보를 처리할 수 있는 모델로 확장
- CNN + LSTM   
- CNN + Transformer

### 📁 Data Structure
```
CAT_image_2nd/
├── 20201028_cat-arch-000156.mp4/
│   ├── f20201028_cat-arch-000156.mp4.json     # 해당 영상 메타데이터 (프레임별 timestamp, keypoints, bbox 등)
│   ├── frame_12_timestamp_800.jpg
│   └── ... (프레임 이미지들)
├── cat-armstretch-080706/    
│   ├── cat-armstretch-080706.json
│   ├── frame_0.jpg
│   ├── frame_1.jpg
│   └── ... (프레임 이미지들)
└── ~    

```
### 📁 dataset.py
```
CatVideoDataset
 ├─ __getitem__ : (T, C, H, W) 텐서 반환
 ├─ frame 부족 시 padding
 ├─ action/emotion/situation 라벨 인코딩
 └─ PIL → Tensor 변환 transform 적용

get_dataset()
 └─ config로부터 label map, transform 구성 후 dataset 반환

collate_fn()
 └─ batch 단위 텐서 묶기 (frames, 3가지 라벨)
```

