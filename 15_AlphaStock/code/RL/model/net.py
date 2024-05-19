import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
import math
from tqdm import tqdm
import numpy as np
'''
AlphaStock
'''


class LSTM_HA(nn.Module):
    '''
    Here we employ the attention to LSTM to capture the time series traits more efficiently.
    '''
    def __init__(self, in_features,
                 hidden_dim = 64,
                 output_dim = 64,
                 n_heads=4,):
        super(LSTM_HA, self).__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=output_dim,batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first= True)
    def forward(self, x):
        e = []
        for i in range(x.shape[0]):
            outputs, (h_n, c_n) = self.lstm(x[i])
            h = self.attention(outputs, outputs, outputs)
            e.append(h[0][:,-1,:])
        e = torch.stack(e, dim = 0)
        return e.relu()
class CAAN(nn.Module):
    """Cross-Asset Attention"""
    def __init__(self, input_dim, output_dim, hidden_dim = 64, n_heads=4):
        super().__init__()

        self.W_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, output_dim)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first= True)

    def forward(self, x):
        queries, keys, values = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = self.W_s(self.attention(   
                                        queries,
                                        keys,
                                        values,
                                        )[0]
                            )
        return scores.relu()
class AlphaStock(torch.nn.Module):
    def __init__(self, dim_in, dim_enc, n_heads, negative_slope):
        super(AlphaStock, self).__init__()
        
        self.lstm_ha = LSTM_HA(dim_enc, dim_enc, dim_enc, n_heads)
        self.caan = CAAN(dim_enc, dim_enc, dim_enc, n_heads)

        #Dense layers for managing network inputs and outputs
        self.input_fc = nn.Linear(dim_in, dim_enc)
        self.out_fc = nn.Linear(dim_enc, 1)

        self.leakyrelu = nn.LeakyReLU(negative_slope)
    
    def forward(self, x):
        e = self.input_fc(x)
        e = self.leakyrelu(e)
        e = self.lstm_ha(e)
        e = self.caan(e)

        return self.out_fc(e).flatten(start_dim = -2)

class PPO(torch.nn.Module):
    def __init__(self, dim_in, dim_enc, n_enc_layers, n_heads, negative_slope, num_stocks = 255, device='cuda:1'):
        super(PPO, self).__init__()

        self.leakyrelu = nn.LeakyReLU(negative_slope)

        self.ln_1 = nn.LayerNorm(dim_enc,elementwise_affine=True)
        self.ln_2 = nn.LayerNorm(dim_enc,elementwise_affine=True)
        self.ln_3 = nn.LayerNorm(dim_enc,elementwise_affine=True)

        self.fc_atten = nn.Linear(dim_enc, dim_enc)
        self.fc_in = nn.Linear(dim_in, dim_enc)
        self.fc_out = nn.Linear(dim_enc, 1)

        
    def forward(self, x ):
        #[batch, num_stocks, window_len, in_features]
        e = self.fc_in(x[:, :, -1, :])
        e = self.leakyrelu(e)
        e = self.ln_1(e)

        e = self.fc_atten(e)
        e = self.leakyrelu(e)
        e = self.ln_2(e)

        e = self.fc_out(e)
        return e.flatten(-2)



class ActorSoftmax(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_encoder_layers=2, n_heads=2, negative_slope=0.1, num_stocks=255):
        super(ActorSoftmax, self).__init__()
        self.actor = AlphaStock(dim_in=input_dim, dim_enc=hidden_dim, n_heads=n_heads, negative_slope=negative_slope)
    def forward(self,x):
        x = self.actor(x)
        probs = F.softmax(x,dim=1)
        return probs
class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_encoder_layers=2, n_heads=2, negative_slope=0.1, num_stocks=255):
        super(Critic,self).__init__()
        assert output_dim == 1 # critic must output a single value
        self.critic = AlphaStock(dim_in=input_dim, dim_enc=hidden_dim, n_heads=n_heads, negative_slope=negative_slope)
        self.fc = nn.Linear(num_stocks, output_dim)
    def forward(self,x):
        x = self.critic(x)
        value = self.fc(x)
        return value
    

if __name__ == "__main__":
    a = torch.rand(2,255, 20,6)
    dim_in = 6
    dim_enc = 64
    n_heads = 4
    negative_slope = 1
    n_encoder_layers = 1
    model = PPO(dim_in=dim_in, dim_enc=dim_enc, n_enc_layers=n_encoder_layers, n_heads=n_heads, negative_slope=negative_slope, num_stocks=255)
    