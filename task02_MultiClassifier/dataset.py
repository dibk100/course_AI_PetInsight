import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from utils import *

class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, transform=None, label_maps=None):
        """
        Args:
            csv_path (str): 라벨 및 이미지 경로가 적힌 CSV 파일 경로
            transform (callable, optional): 이미지 전처리 함수
            label_maps (dict): {'action': dict, 'emotion': dict, 'situation': dict}
                               라벨명 -> 인덱스 매핑 딕셔너리
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.label_maps = label_maps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 문자열 라벨 → 인덱스 변환
        action_label = self.label_maps['action'][row['action']]
        emotion_label = self.label_maps['emotion'][row['emotion']]
        situation_label = self.label_maps['situation'][row['situation']]

        # tensor(int) 형태로 반환
        return image, torch.tensor(action_label, dtype=torch.long), torch.tensor(emotion_label, dtype=torch.long), torch.tensor(situation_label, dtype=torch.long)

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