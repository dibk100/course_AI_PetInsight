import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, video_dirs, label_encoder, transform=None, max_frames=100):
        self.video_dirs = video_dirs          # 각 동영상 폴더 경로 리스트
        self.label_encoder = label_encoder
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_path = self.video_dirs[idx]

        # --- 1. Load JSON
        json_file = [f for f in os.listdir(video_path) if f.endswith(".json")]
        if not json_file:
            raise FileNotFoundError(f"JSON not found in {video_path}")
        with open(os.path.join(video_path, json_file[0]), "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- 2. Extract labels & encode
        meta = data["metadata"]
        inspect = meta.get("inspect", {})
        owner = meta.get("owner", {})

        action_str = inspect.get("action") or meta.get("action")
        emotion_str = inspect.get("emotion") or owner.get("emotion")
        situation_str = owner.get("situation")

        try:
            action_label = self.label_encoder["action"][action_str]
            emotion_label = self.label_encoder["emotion"][emotion_str]
            situation_label = self.label_encoder["situation"][situation_str]
        except KeyError as e:
            raise ValueError(f"Label not found in encoder: {e}")

        # --- 3. Load image frames
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])
        frame_files = frame_files[:self.max_frames]
        images = []

        for fname in frame_files:
            img_path = os.path.join(video_path, fname)
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            images.append(img)

        # --- 4. Pad if too short
        T = len(images)
        if T < self.max_frames:
            pad_tensor = torch.zeros_like(images[0])
            for _ in range(self.max_frames - T):
                images.append(pad_tensor)

        frames = torch.stack(images)  # [T, C, H, W]
        labels = torch.tensor([action_label, emotion_label, situation_label], dtype=torch.long)

        return frames, labels

def get_dataset(config, split='train'):
    """
    config에 지정된 CSV 경로와 transform, label_maps를 사용해
    split별 Dataset 객체를 반환하는 함수

    split: 'train', 'val', 'test' 중 하나
    """
    csv_path = config['data'][f'{split}_csv']  # 예: config['data']['train_csv'] = './data/train.csv'
    label_maps = config.get('label_maps', None)
    
    # 기본 transform, 필요시 config에 추가 가능
    from torchvision import transforms
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:  # val, test
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    dataset = MultiLabelDataset(csv_path, transform=transform, label_maps=label_maps)
    return dataset