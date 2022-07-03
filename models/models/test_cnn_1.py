import torch
import torch.nn as nn
import torch.nn.functional as F

class TestNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.convn2 = nn.Conv2d(128, 64, 3)
        self.convn1 = nn.Conv2d(64,32,3)
        
        num_out_nur = 32
        self.fc1 = nn.Linear(num_out_nur , num_out_nur//4)
        self.fc2 = nn.Linear(num_out_nur//4 , num_out_nur//16)
        self.fc3 = nn.Linear(num_out_nur//16, 2)

    def convs(self, x):
        x = F.max_pool2d(
            F.relu(self.conv1(x)),
            (3,3)
            )
        
        x = F.max_pool2d(
            F.relu(self.conv2(x)),
            (3,3)
            )
        
        x = F.max_pool2d(
            F.relu(self.conv3(x)),
            (3,3)
            )

        
        x = F.max_pool2d(
            F.relu(self.convn2(x)),
            (2,2)
            )
        
        x = F.max_pool2d(
            F.relu(self.convn1(x)),
            (2,2)
            )

        return x

    def forward(self, x):
        x = self.convs(x)
        x = F.relu(self.fc1(torch.flatten(x,1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim = 1)