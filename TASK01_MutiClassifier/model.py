import torch
import torch.nn as nn
import torchvision.models as models
import timm

# timm.list_models(pretrained=True)             # 모델 목록 확인

class MultiLabelImageClassifier(nn.Module):
    def __init__(self, num_actions, num_emotions, num_situations,
                 backbone_name='resnet18', pretrained=True):
        super(MultiLabelImageClassifier, self).__init__()

        # 마지막 FC 제거하고 feature extractor만 추출
        # self.shared_encoder = nn.Sequential(*list(base_model.children())[:-1])  # classification head 제거
        # self.feature_dim = base_model.fc.in_features                            # 마지막 FC layer가 사용하던 입력 feature 수 (512 or 2048, 모델에 따라 다름)를 가져오는 코드
        
        self.shared_encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)  # 🔸 classification head 제거
        self.feature_dim = self.shared_encoder.num_features  # 🔸 마지막 feature 차원 추출

        # 다중 분류기 head
        self.action_head = nn.Linear(self.feature_dim, num_actions)
        self.emotion_head = nn.Linear(self.feature_dim, num_emotions)
        self.situation_head = nn.Linear(self.feature_dim, num_situations)

    def forward(self, x):
        x = self.shared_encoder(x)          # (B, C, 1, 1)            ## backbone or shared_encoder
        x = x.view(x.size(0), -1)           # (B, C)

        # 각 head 출력
        action_logits = self.action_head(x)
        emotion_logits = self.emotion_head(x)
        situation_logits = self.situation_head(x)
        return action_logits, emotion_logits, situation_logits
    
    def get_model(self):
        return self
