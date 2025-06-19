import torch
import torch.nn as nn
import timm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiLabelVideoTransformerClassifier(nn.Module):
    def __init__(self, num_actions, num_emotions, num_situations,
                 backbone_name='resnet18', pretrained=True,
                 trans_layers=4, trans_heads=8, trans_dim=512):
        super().__init__()

        # 1. CNN 백본
        self.shared_encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.shared_encoder.num_features  # 예: 512

        # 2. 포지셔널 인코딩
        self.pos_encoding = PositionalEncoding(d_model=self.feature_dim)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim,
                                                   nhead=trans_heads,
                                                   dim_feedforward=2048,
                                                   dropout=0.1,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # 4. 분류 헤드
        self.action_head = nn.Linear(self.feature_dim, num_actions)
        self.emotion_head = nn.Linear(self.feature_dim, num_emotions)
        self.situation_head = nn.Linear(self.feature_dim, num_situations)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.shared_encoder(x)  # [B*T, feature_dim]
        feats = feats.view(B, T, -1)    # [B, T, feature_dim]

        feats = self.pos_encoding(feats)      # positional encoding 추가
        encoded = self.transformer(feats)     # [B, T, feature_dim]

        pooled = encoded.mean(dim=1)          # [B, feature_dim] (평균 pooling)

        # 멀티 분류 head
        return (
            self.action_head(pooled),
            self.emotion_head(pooled),
            self.situation_head(pooled)
        )
