import torch
from torch import nn
from torchmetrics import F1Score, AUROC

class LandmarkDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(LandmarkDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
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
    def __init__(self, feature_size=136, lm_dropout_rate=0.1, rnn_unit=32,
                 num_layers=1, rnn_dropout_rate=0,
                 fc_dropout_rate=0.5, res_hidden=64, num_heads=4, cnn_channels=64, kernel_size=3,
                 ablation_mode='mha'):  # 设置 ablation_mode='mha'，去除多头注意力层
        super(LRNet, self).__init__()
        self.hidden_size = rnn_unit
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ablation_mode = ablation_mode

        # Landmark Dropout Layer
        self.dropout_landmark = LandmarkDropout(lm_dropout_rate)

        # Convolutional Layer for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=feature_size, out_channels=cnn_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # GRU Layer
        self.gru = nn.GRU(input_size=cnn_channels, hidden_size=rnn_unit,
                          num_layers=num_layers, dropout=rnn_dropout_rate,
                          batch_first=True, bidirectional=True)

        # LSTM Layer for additional long-sequence modeling
        self.lstm = nn.LSTM(input_size=rnn_unit * 2, hidden_size=rnn_unit,
                            num_layers=num_layers, dropout=rnn_dropout_rate,
                            batch_first=True, bidirectional=True)

        # Feed Forward with Residual Connection
        self.feed_forward = nn.Sequential(
            Residual(FeedForward(rnn_unit * 2, res_hidden, fc_dropout_rate)),
            nn.Dropout(fc_dropout_rate),
        )

        # Dense Layer with Residual Block and Dropout
        self.dense = nn.Sequential(
            nn.Linear(rnn_unit * 2, 2)
        )

        # Output Softmax Layer
        self.output = nn.Softmax(dim=1)

        # F1 and AUC metrics
        self.auc_metric = AUROC(task="binary", num_classes=2)
        self.f1_metric = F1Score(task="binary", num_classes=2)

    def forward(self, x):
        # Apply Landmark Dropout
        x = self.dropout_landmark(x)

        # Reshape for CNN layer: (batch, features, frames)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Reshape back to (batch, frames, features)

        # Initial hidden state for GRU and LSTM
        h0_gru = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size, device=x.device)
        h0_lstm = (torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size, device=x.device),
                   torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size, device=x.device))

        # Pass through GRU Layer
        gru_out, _ = self.gru(x, h0_gru)

        # Pass through LSTM Layer
        lstm_out, _ = self.lstm(gru_out, h0_lstm)

        # 跳过多头注意力层
        attn_out = lstm_out  # Skip MHA, directly use LSTM output

        # Feed Forward Network with Residual Connection
        x = self.feed_forward(attn_out[:, -1, :])  # Only take the output of the last timestep

        # Dense and Softmax Output
        x = self.dense(x)
        x = self.output(x)

        return x

    def compute_metrics(self, predictions, labels):
        # Compute F1 Score
        f1 = self.f1_metric(predictions, labels)
        # Compute AUC
        auc = self.auc_metric(predictions, labels)
        return {'F1 Score': f1.item(), 'AUC': auc.item()}
