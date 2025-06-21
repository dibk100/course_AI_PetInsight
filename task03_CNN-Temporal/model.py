import torch
import torch.nn as nn
import timm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=160):       # 프레임 최대 150
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

        # 3. Transformer Encoder :: Transformer 출력 후 LayerNorm + Dropout
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim,
                                                   nhead=trans_heads,
                                                   dim_feedforward=2048,
                                                   dropout=0.1,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # 4. 분류 헤드 : Multi-task head 간 간섭 가능성 때문에 수정함
        # self.action_head = nn.Linear(self.feature_dim, num_actions)
        # self.emotion_head = nn.Linear(self.feature_dim, num_emotions)
        # self.situation_head = nn.Linear(self.feature_dim, num_situations)
        
        self.action_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, num_actions)
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, num_emotions)
        )
        
        self.situation_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, num_situations)
        )
        
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(self.feature_dim)
        
        self.attention_pooling = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.shared_encoder(x)  # [B*T, feature_dim]
        feats = feats.view(B, T, -1)    # [B, T, feature_dim]

        feats = self.pos_encoding(feats)      # positional encoding 추가
        encoded = self.transformer(feats)     # [B, T, feature_dim]
        # encoded = self.norm(encoded)        # 정규화 : transforemer 내부에 norm이 적용되고 있ㅇㄹ 수 있음
        
        # Attention-based pooling
        # attn_scores = self.attention_pooling(encoded)        # [B, T, 1]
        # attn_weights = torch.softmax(attn_scores, dim=1)     # [B, T, 1]
        # pooled = (encoded * attn_weights).sum(dim=1)         # [B, D]
        
        pooled = encoded.mean(dim=1)           # [B, feature_dim] (평균 pooling) -> 모든 프레임의 feature를 동등하게 평균함. 중요 프레임 구분을 못함.
        pooled = self.dropout(pooled)        # dropout 적용 :: 위치 고민

        # 시각화용으로 반환
        # return self.action_head(pooled), self.situation_head(pooled), attn_weights.squeeze(-1)
        
        # 멀티 분류 head
        return (
            self.action_head(pooled),
            self.emotion_head(pooled),
            self.situation_head(pooled)
        )
        
    def get_model(self):
        return self

class MultiLabelVideoLSTMClassifier(nn.Module):
    def __init__(self, num_actions, num_emotions, num_situations,
                 backbone_name='resnet18', pretrained=True,
                 lstm_hidden_dim=512, lstm_layers=1, dropout=0.2):
        super().__init__()

        # CNN backbone
        self.shared_encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.shared_encoder.num_features

        # LSTM
        self.lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=False)

        self.dropout = nn.Dropout(dropout)

        # Attention pooling (optional)
        self.attention_pooling = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Task heads
        self.action_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim // 2, num_actions)
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim // 2, num_emotions)
        )
        self.situation_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim // 2, num_situations)
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.shared_encoder(x)  # [B*T, feature_dim]
        feats = feats.view(B, T, -1)    # [B, T, feature_dim]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(feats)  # lstm_out: [B, T, hidden_dim]

        # Attention pooling on lstm_out
        # attn_scores = self.attention_pooling(lstm_out)  # [B, T, 1]
        # attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T, 1]
        # pooled = (lstm_out * attn_weights).sum(dim=1)  # [B, hidden_dim]
        pooled = lstm_out.mean(dim=1)

        pooled = self.dropout(pooled)

        return (
            self.action_head(pooled),
            self.emotion_head(pooled),
            self.situation_head(pooled)
        )
    def get_model(self):
        return self
    
MODEL_CLASSES = {
    "transformer": MultiLabelVideoTransformerClassifier,
    "lstm": MultiLabelVideoLSTMClassifier,
}