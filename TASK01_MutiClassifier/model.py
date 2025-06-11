import torch
import torch.nn as nn
import torchvision.models as models

class MultiLabelImageClassifier(nn.Module):
    def __init__(self, num_actions, num_emotions, num_situations,
                 backbone_name='resnet18', pretrained=True):
        super(MultiLabelImageClassifier, self).__init__()

        # 🔸 지원하는 backbone 목록 (확장 가능)
        backbone_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }

        assert backbone_name in backbone_dict, f"Unsupported backbone: {backbone_name}"

        base_model = backbone_dict[backbone_name](pretrained=pretrained)

        # 🔸 마지막 FC 제거하고 feature extractor만 추출
        self.shared_encoder = nn.Sequential(*list(base_model.children())[:-1])  # AdaptiveAvgPool2d 포함
        self.feature_dim = base_model.fc.in_features                            # ResNet의 마지막 FC layer가 사용하던 입력 feature 수 (512 or 2048, 모델에 따라 다름)를 가져오는 코드

        # 🔸 다중 분류기 head
        self.action_head = nn.Linear(self.feature_dim, num_actions)
        self.emotion_head = nn.Linear(self.feature_dim, num_emotions)
        self.situation_head = nn.Linear(self.feature_dim, num_situations)

    def forward(self, x):
        x = self.shared_encoder(x)          # (B, C, 1, 1)
        x = x.view(x.size(0), -1)           # (B, C)

        # 🔸 각 head 출력
        action_logits = self.action_head(x)
        emotion_logits = self.emotion_head(x)
        situation_logits = self.situation_head(x)
        return action_logits, emotion_logits, situation_logits
    
    def get_model(self):
        return self
