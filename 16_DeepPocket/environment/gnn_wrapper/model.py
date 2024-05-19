import torch
import torch_geometric.nn as nn
import numpy as np

class ChebNetwork(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,K):
        super(ChebNetwork, self).__init__()
        self.input_conv = nn.ChebConv(in_channels, hidden_channels[0], K=K).to(torch.float64)
        self.hidden_layers = []
        for i in range(1,len(hidden_channels)):
            self.hidden_layers.append(nn.ChebConv(hidden_channels[i-1], hidden_channels[i], K=K).to(torch.float64))
        self.output_conv = nn.ChebConv(hidden_channels[len(hidden_channels)-1], out_channels, K=K).to(torch.float64)

    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = torch.sigmoid(self.input_conv(x, edge_index, edge_weight))
        for layer in self.hidden_layers:
            x = torch.sigmoid(layer(x, edge_index, edge_weight))

        return torch.sigmoid(self.output_conv(x, edge_index, edge_weight))
        