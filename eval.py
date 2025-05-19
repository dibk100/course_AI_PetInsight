from sklearn.metrics import classification_report
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

# 데이터 로더
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
test_data = datasets.ImageFolder(root='data/', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 모델 불러오기
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load("cat_behavior_model.pth"))
model.to(DEVICE)
model.eval()

# 예측
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        outputs = model(x)
        preds = outputs.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(y.tolist())

print(classification_report(y_true, y_pred, target_names=test_data.classes))
