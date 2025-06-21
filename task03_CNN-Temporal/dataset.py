import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Callable, Dict, Any
import torch

class CatVideoDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        label2idx_action: Optional[Dict[str, int]] = None,
        label2idx_emotion: Optional[Dict[str, int]] = None,
        label2idx_situation: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
        max_frames: int = 100,
    ):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames

        self.label2idx_action = label2idx_action or {label: i for i, label in enumerate(self.df['action'].unique())}
        #self.label2idx_emotion = label2idx_emotion or {label: i for i, label in enumerate(self.df['emotion'].unique())}
        self.label2idx_situation = label2idx_situation or {label: i for i, label in enumerate(self.df['situation'].unique())}

        self.idx2label_action = {v: k for k, v in self.label2idx_action.items()}
        #self.idx2label_emotion = {v: k for k, v in self.label2idx_emotion.items()}
        self.idx2label_situation = {v: k for k, v in self.label2idx_situation.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        video_folder = os.path.join(self.root_dir, row['video_name'])
        frame_count = row['frames']

        label_action = self.label2idx_action[row['action']]
        label_situation = self.label2idx_situation[row['situation']]

        frames = []
        for i in range(min(self.max_frames, frame_count)):
            frame_path = os.path.join(video_folder, f"{i+1:06d}.jpg")
            if os.path.exists(frame_path):
                image = Image.open(frame_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                frames.append(image)

        # 패딩 부분 수정
        if len(frames) < self.max_frames:
            if frames:
                pad_frame = torch.zeros_like(frames[0])  # frames[0]과 동일한 크기와 타입의 0 tensor 생성
            else:
                # frames가 하나도 없으면 새로 0 tensor 생성 (3채널, 224x224)
                pad_frame = torch.zeros(3, 224, 224)
            while len(frames) < self.max_frames:
                frames.append(pad_frame)

        frames_tensor = torch.stack(frames)  # (T, C, H, W)

        return {
            "video_name": row['video_name'],
            "frames": frames_tensor,
            "label_action": label_action,
            "label_situation": label_situation
        }

def get_dataset(config: dict, split: str = 'train') -> Dataset:
    csv_path = config['data'][f'{split}_csv']
    root_dir = config['data'].get('root_dir', './frames')
    max_frames = config.get('max_frames', 100)

    label_maps = {
        'action': {label: i for i, label in enumerate(config['label_names']['action'])},
        #'emotion': {label: i for i, label in enumerate(config['label_names']['emotion'])},
        'situation': {label: i for i, label in enumerate(config['label_names']['situation'])},
    }

    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    dataset = CatVideoDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        label2idx_action=label_maps['action'],
        #label2idx_emotion=label_maps['emotion'],
        label2idx_situation=label_maps['situation'],
        transform=transform,
        max_frames=max_frames,
    )
    return dataset

def collate_fn(batch):
    frames = torch.stack([item['frames'] for item in batch])  # (B, T, C, H, W)
    labels_action = torch.tensor([item['label_action'] for item in batch])
    labels_situation = torch.tensor([item['label_situation'] for item in batch])
    return frames, labels_action, labels_situation
