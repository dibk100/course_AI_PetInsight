import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

def get_model(model_name: str, num_classes: int = 6):
    """
    모델 이름에 따라 모델 객체를 반환

    Args:
        model_name (str): 'resnet18', 'resnet50', 'custom_resnet50'
        num_classes (int): 출력 클래스 수

    Returns:
        nn.Module: 선택한 모델 객체
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True,weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'custom_resnet50':
        model = CustomResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet50, self).__init__()

        # 사전 학습된 resnet50 불러오기 (특징 추출기)
        base_model = models.resnet50(pretrained=True)

        # 초기 레이어들 (conv1 ~ layer4까지 그대로 사용?) :: test
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])  # 마지막 fc, avgpool 제외

        # 커스텀 레이어 추가 (conv, dropout, batchnorm 등)          :: test
        self.custom_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.custom_head(x)
        return x