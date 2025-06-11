from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MultiLabelImageDataset(Dataset):
    def __init__(self, image_paths, action_labels, emotion_labels, situation_labels, transform=None):
        self.image_paths = image_paths
        self.action_labels = action_labels
        self.emotion_labels = emotion_labels
        self.situation_labels = situation_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, self.action_labels[idx], self.emotion_labels[idx], self.situation_labels[idx]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])