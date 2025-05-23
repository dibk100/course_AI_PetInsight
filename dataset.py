import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

class CatBehaviorDataset(Dataset):
    def __init__(self, json_dir, img_root_dir, transform=None, task='action',describe = False):
        """
        Args:
            json_dir (str): JSON 파일들이 있는 CAT_labeled 디렉토리
            img_root_dir (str): 이미지가 있는 CAT_raw 디렉토리
            transform (callable, optional): 이미지에 적용할 torchvision transform
            task (str): 'action' 또는 'emotion' 중 하나
        """
        assert task in ['action', 'emotion'], "task는 'action' 또는 'emotion' 이어야 합니다."
        self.json_dir = json_dir
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.task = task
        self.samples = []  # (image_path, label) 리스트
        self.describe = describe        # 파일 상태 확인하고 싶을 때 True

        self._load_annotations()

    def _load_annotations(self):
        total_video = 0
        total_image = 0
        missing_file = 0
        pass_file = 0
        skip_file = 0
        
        for json_file in os.listdir(self.json_dir):
            total_video += 1
            if not json_file.endswith('.json'):
                print("[Pass] json file : ",json_file)
                pass_file +=1
                continue

            json_path = os.path.join(self.json_dir, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            video_path = data.get('file_video')
            if not video_path:
                print("[Pass] file_video : ",json_file)
                pass_file +=1
                continue
            
            video_file = os.path.basename(video_path)  # cat-armstretch-011278.mp4
            video_name = os.path.splitext(video_file)[0]  # cat-armstretch-011278
            parent_dir = os.path.dirname(video_path)     # 20210105
            full_video_folder = f"{parent_dir}_{video_file}"  # 20210105_cat-armstretch-011278.mp4
            
            # 여러 후보 경로 생성
            candidate_folders = [
                os.path.join(self.img_root_dir, full_video_folder),  # 20210105_cat-armstretch-011278.mp4
                os.path.join(self.img_root_dir, video_name),         # cat-armstretch-011278
            ]
            
            found_folder = None
            for folder in candidate_folders:
                if os.path.exists(folder):
                    found_folder = folder
                    break
                        
            if found_folder is None:
                print(f"[SKIP] Folder not found for {video_path}")
                skip_file +=1
                continue
            
            inspect_meta = data['metadata'].get('inspect', {})
            label = inspect_meta.get(self.task)
            if label is None:
                continue

            for ann in data.get('annotations', []):
                total_image +=1
                frame_num = ann['frame_number']
                timestamp = ann['timestamp']
                img_name = f"frame_{frame_num}_timestamp_{timestamp}.jpg"
                img_path = os.path.join(found_folder, img_name)

                if os.path.exists(img_path):
                    self.samples.append((img_path, label))
                else:
                    # print(f"[MISSING] {img_path}")
                    missing_file +=1
        
        if self.describe :
            print("동영상 수 : ",total_video)  
            print("프레임 수 : ",total_image)     
            print("미씽 프레임 수 : ",missing_file)
            print("패스 파일 수 : ",pass_file)
            print("스킵 폴더 수 : ",skip_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transform():
    return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


def get_dataloaders(batch_size=32, task='emotion', json_dir='data_mini_cat/CAT_labeled', img_root_dir='data_mini_cat/CAT_raw'):
    transform = get_transform()
    dataset = CatBehaviorDataset(transform = transform,json_dir=json_dir, img_root_dir=img_root_dir, task=task)

    # 전체 데이터 개수
    total_size = len(dataset)
    if total_size == 0:
        raise ValueError("Dataset is empty. Check if the paths and label extraction are correct.")

    # 학습/검증 비율 (예: 80/20)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader