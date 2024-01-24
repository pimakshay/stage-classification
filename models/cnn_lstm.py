from torch import nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_channels, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.cnnetwork = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size = 3, padding = 1), # o/p: 150x150x32 (img_size: 150x150x3)
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1), # o/p: 150x150x64
            nn.ReLU(),
            nn.MaxPool2d(2,2), # o/p 75x75x64
        )
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        cnnout = self.cnnetwork(x)
        out, _ = self.lstm(cnnout.view(cnnout.size(0), -1)) # returns pred, lstm_cell_memory, lstm_hidden_states
        out = self.fc(out)#self.fc(out[:, -1, :])
        return out