import torch
from torch import nn

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
    def __init__(self, feature_size=136, lm_dropout_rate=0.1, rnn_unit=32,
                 num_layers=1, rnn_dropout_rate=0, fc_dropout_rate=0.5, res_hidden=64, cnn_channels=64):
        super(LRNet, self).__init__()
        self.hidden_size = rnn_unit
        self.num_layers = num_layers

        # Landmark Dropout Layer
        self.dropout_landmark = LandmarkDropout(lm_dropout_rate)

        # CNN Layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=feature_size, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # GRU Layer
        self.gru = nn.GRU(input_size=cnn_channels, hidden_size=rnn_unit,
                          num_layers=num_layers, dropout=rnn_dropout_rate,
                          batch_first=True, bidirectional=True)

        # Fully Connected Layers
        self.dense = nn.Sequential(
            nn.Dropout(fc_dropout_rate),
            Residual(FeedForward(rnn_unit * 2, res_hidden, fc_dropout_rate)),
            nn.Dropout(fc_dropout_rate),
            nn.Linear(res_hidden, 2)  # Final classification layer
        )
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        # Apply Landmark Dropout
        x = self.dropout_landmark(x)

        # CNN Layer expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # Reshape to (batch, features, frames) for CNN
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Reshape back to (batch, frames, features) for RNN

        # GRU expects input as (batch, seq_len, features)
        _, hidden = self.gru(x)

        # Concatenate GRU hidden states from both directions
        x = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, rnn_unit * 2)

        # Fully connected layer
        x = self.dense(x)
        x = self.output(x)
        return x
