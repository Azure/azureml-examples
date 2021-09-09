# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# define network(s)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.l1 = nn.Linear(4, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.softmax(self.l3(x), dim=1)

        return x
