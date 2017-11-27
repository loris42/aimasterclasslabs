import torch.nn as nn
import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc0 = nn.Linear(28*28, 27)

#     def forward(self, x):
#         x = self.fc0(x.view(x.size(0), -1))
        
#         return F.log_softmax(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.conv2 = nn.Conv2d(4, 10, 5)
        self.fc1 = nn.Linear(4*4*10, 500)
        self.fc2 = nn.Linear(500, 27)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*10)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x)
