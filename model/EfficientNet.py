import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class LandmarkDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(LandmarkDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def generate_mask(self, landmark, frame):
        position_p = torch.bernoulli(torch.Tensor([1 - self.p] * (landmark // 2)))
        position_p = position_p.repeat_interleave(2)
        return position_p.repeat(1, frame, 1)

    def forward(self, x: torch.Tensor):
        if self.training:
            _, frame, landmark = x.size()
            landmark_mask = self.generate_mask(landmark, frame)
            scale = 1 / (1 - self.p)
            return x * landmark_mask.to(x.device) * scale
        else:
            return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class LRNet(nn.Module):
    def __init__(self, lm_dropout_rate=0.1, fc_dropout_rate=0.5, res_hidden=64):
        super(LRNet, self).__init__()

        # Landmark Dropout Layer
        self.dropout_landmark = LandmarkDropout(lm_dropout_rate)

        # EfficientNet Layer for Forgery Detection (Feature Extraction)
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b0')

        # Fully Connected Layers for Classification
        self.dense = nn.Sequential(
            nn.Dropout(fc_dropout_rate),
            Residual(FeedForward(1280, res_hidden, fc_dropout_rate)),  # EfficientNet-b0 has 1280 output features
            nn.Dropout(fc_dropout_rate),
            nn.Linear(res_hidden, 2)  # Final classification layer
        )
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        # Apply Landmark Dropout
        x = self.dropout_landmark(x)

        # EfficientNet expects (batch, channels, height, width)
        x = self.feature_extractor.extract_features(x)  # Feature extraction

        # Flatten the output for the fully connected layers
        x = x.mean([2, 3])  # Global average pooling over spatial dimensions

        # Fully connected layer
        x = self.dense(x)
        x = self.output(x)
        return x
