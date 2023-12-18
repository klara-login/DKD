import time
import torch
import torch.nn as nn
import numpy as np

class WDCNN(nn.Module):
    
    def __init__(self):
        super(WDCNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,16,64,16,24),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16,32,3,1,1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32,64,3,0,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64,64,3,0,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(64,64,3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,2)
        )
        self.layer6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
