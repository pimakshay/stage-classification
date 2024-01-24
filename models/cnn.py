import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    # _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class CNNClassification(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        ## using http://layer-calc.com/ to calculate output dimensions
        self.network = nn.Sequential(
            
            nn.Conv2d(input_channels, 32, kernel_size = 3, padding = 1), # o/p: 150x150x32 (img_size: 150x150x3)
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1), # o/p: 150x150x64
            nn.ReLU(),
            nn.MaxPool2d(2,2), # o/p 75x75x64
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1), # o/p: 75x75x128
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1), # o/p: 75x75x128
            nn.ReLU(),
            nn.MaxPool2d(2,2), # o/p: 37x37x128
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1), # o/p: 37x37x256
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1), # o/p: 37x37x256
            nn.ReLU(),
            nn.MaxPool2d(2,2), # o/p: 18x18x256  (for img_size: 224 --> (28x28x256))
            
            nn.Flatten(), # o/p: 82944x1
            nn.Linear(82944,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,num_classes)
        )
    
    def forward(self, xb):
        return self.network(xb)