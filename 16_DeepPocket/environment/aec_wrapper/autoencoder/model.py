import torch.nn as nn
import torch

class RAESC(nn.Module):
    def __init__(self,input_channels,hidden_channels,output_channels, seq_len):
        super(RAESC,self).__init__()
        self.seq_len = seq_len
        self.rnn1 = nn.LSTM(input_channels,output_channels,batch_first=True)
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels

        self.conv1 = nn.Conv1d(self.output_channels,self.hidden_channels,kernel_size=3,padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.rnn2 = nn.LSTM(self.hidden_channels,self.output_channels)
        self.lin =  nn.Linear(self.output_channels,self.output_channels)
        
    def encode(self,x):
        return self.rnn1(x)
    
    def decode(self, x):
        x = x[:,-1,:]
        x = x.reshape((-1,self.output_channels, 1))
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.reshape((-1,1,self.hidden_channels))
        x,_ = self.rnn2(x)
        x = x[:, -1, :]
        x = self.lin(x)

        return x

    def forward(self,x):
        x,_ = self.encode(x)
        return self.decode(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Conv1d(1,5,kernel_size=(3,), stride=(2,)),
            nn.ReLU(),
            nn.Conv1d(5,10,kernel_size=(3,), stride=(1,)),
            nn.ReLU(),
            nn.Conv1d(10,3,kernel_size=(3,), stride=(1,)),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(3,10,kernel_size=(3,),stride=(1,)),
            nn.ReLU(),
            nn.ConvTranspose1d(10,5,kernel_size=(3,),stride = (1,)),
            nn.ReLU(),
            nn.ConvTranspose1d(5,1,kernel_size=(3,), stride = (2,))
        )

    def forward(self, x):
        x = self.encode(x)
        x = x.reshape((-1,3, 1))
        x = self.decoder(x)

        return x.reshape((-1,11,1)).squeeze(-1)

class LinearAutoEncoder(nn.Module):
    def __init__(self,in_features,hidden_size,out_features):
        super(LinearAutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
                                    nn.Linear(in_features, hidden_size[0]).to(torch.float64), nn.ReLU(),
                                    nn.Linear(hidden_size[0], hidden_size[1]).to(torch.float64), nn.ReLU(),
                                    nn.Linear(hidden_size[1], out_features).to(torch.float64)
                                    ) 

        self.decoder = nn.Sequential(
                                    nn.Linear(out_features, out_features).to(torch.float64), nn.ReLU(),
                                    nn.Linear(out_features, out_features).to(torch.float64), nn.ReLU(),
                                    nn.Linear(out_features, out_features).to(torch.float64)
                                    )

        self.out_features = out_features

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self,x):
        x = self.encode(x)
        return self.decode(x).reshape(-1, self.out_features, 1).squeeze(-1) 
              