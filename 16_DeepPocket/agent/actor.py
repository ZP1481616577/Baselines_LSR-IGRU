import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):

    def __init__(self,in_channels, trading_window_size,actor_lr, gnn_parameters,actor_weight_decay):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,3,kernel_size=(1,3)).to(torch.float64)
        self.conv2 = nn.Conv2d(3,3, kernel_size=(1,trading_window_size-2)).to(torch.float64)
        self.conv3 = nn.Conv2d(4,1,kernel_size=(1,1)).to(torch.float64)
        #params = list(self.parameters()) + list(gnn_parameters) 
        self.optimizer = optim.Adam(self.parameters(),lr = actor_lr,weight_decay = actor_weight_decay)

    def forward(self, x, prev_weigths):
        prev_weigths = prev_weigths.clone().detach()
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.cat((prev_weigths[1:].unsqueeze(-1).unsqueeze(0),x.squeeze(0))).unsqueeze(0)
        x = torch.tanh(self.conv3(x))
        x = torch.cat((prev_weigths[0].unsqueeze(0).unsqueeze(-1),x.squeeze(0).squeeze(0)))
        return torch.softmax(x,dim = 0).reshape(-1)

        

