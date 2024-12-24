import torch
import torch.nn as nn

class QuickMLP(nn.Module):
    def __init__(self):
        super(QuickMLP, self).__init__()
        self.fc = nn.Linear(768, 256)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x
