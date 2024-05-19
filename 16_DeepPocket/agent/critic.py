import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class Critic(nn.Module):

    def __init__(self,in_channels,num_assets,trading_window_size,lr, weight_decay):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1)).to(torch.float64)
        self.conv2 = nn.Conv2d(in_channels,trading_window_size,kernel_size=(1,1)).to(torch.float64)
        self.conv3 = nn.Conv2d(trading_window_size,1,kernel_size=(1,trading_window_size)).to(torch.float64)
        self.dense = nn.Linear(num_assets,1).to(torch.float64)
        self.optimizer = optim.Adam(self.parameters(),lr = lr,weight_decay= weight_decay)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dense(x.squeeze(-1))
        
        return x.reshape(-1)
