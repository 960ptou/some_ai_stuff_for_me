import torch
import torch.nn as nn
import torch.nn.functional as F

class TestNet2(nn.Module):
    def __init__(self):
        super.__init__()
        self.conv1 = nn.Conv2d(3,32,4)
        self.conv2 = nn.Conv2d(32,64,4)
        
        num_out_nur = 32
        self.fc1 = nn.Linear(num_out_nur, num_out_nur//2)
        self.fc2 = nn.Linear(num_out_nur//2, 1)
        
    def convs(self,x):
        x = F.max_pool2d(
            F.relu(self.conv1(x)),
            (3,3)
        )

        x = F.max_pool2d(
            F.relu(self.conv2(x)),
            (3,3)
        )
        return x
        
    def forward(self, x):
        x = self.convs(x)
        x = F.relu(self.fc1(torch.flatten(x,1)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)