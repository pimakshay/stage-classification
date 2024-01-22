from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size=28, num_classes=7):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        r_out, (_, _) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out