import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from stock_dataset import StockDataset
from model import LinearAutoEncoder
import numpy as np
import matplotlib.pyplot as plt


def train(model):
    # train_stock_skip = ['AAPL.pkl', 'ADBE.pkl', 'ADI.pkl', 'ADP.pkl', 'ADSK.pkl', 'ALGN.pkl', 'AMD.pkl']
    train_stock_skip = ['A.pkl', 'AAL.pkl', 'AAPL.pkl', 'ABBV.pkl', 'ACN.pkl', 'ADBE.pkl', 'AMD.pkl']
    # train_stock_skip = ['000001.SZ.pkl', '000002.SZ.pkl', '000063.SZ.pkl', '000069.SZ.pkl', '000100.SZ.pkl',
     #                   '000157.SZ.pkl', '000166.SZ.pkl']
    # train_stock_skip = ['000009.SZ.pkl', '000021.SZ.pkl', '000027.SZ.pkl', '000037.SZ.pkl', '000039.SZ.pkl',
      #                  '000089.SZ.pkl', '000156.SZ.pkl']

    train_dataset = StockDataset(folder_path = 'E:/baselines/DeepPocket-master/data/sp500',skip_names = train_stock_skip)

    optimizer = optim.Adam(model.parameters(), lr = 0.00001, weight_decay=1e-8)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for epoch in range(30):
        model.train()
        for x,y in train_loader:
            optimizer.zero_grad()
            predicted = model(x)
            loss = criterion(predicted,y)
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print('Epoch: {}, Loss: {:.15f}'.format(epoch, loss.item()))
    

def test(model):
    test_stock_skip = ['A.pkl', 'AAL.pkl', 'AAPL.pkl', 'ABBV.pkl', 'ACN.pkl', 'ADBE.pkl', 'AMD.pkl']
    test_dataset = StockDataset(folder_path='E:/baselines/DeepPocket-master/data/sp500',skip_names = test_stock_skip, flag=False)
    test_loader = DataLoader(test_dataset,batch_size = 1, shuffle = False)
    losses = []
    criterion = nn.MSELoss()
    for x,y in test_loader:
        model.eval()
        with torch.no_grad():
            predicted = model(x)
            loss = criterion(predicted,y)
            losses.append(loss.item())

    print('Mean loss on test_dataset:',np.mean(losses))

    return test_loader

def plot(model, test_loader):
    real_high, real_low, real_close = [], [], []
    predicted_high, predicted_low, predicted_close = [], [], []

    index = 0
    for x,y in test_loader:
        with torch.no_grad():
            output = model(x)
            output = output.detach().numpy()
            y = y.numpy()
            predicted_high.extend(output[:,0])
            predicted_low.extend(output[:,1])
            predicted_close.extend(output[:,2])

            real_high.extend(y[:,0])
            real_low.extend(y[:,1])
            real_close.extend(y[:,2])
            if index == 50:
                break
            index +=1

    X = range(0,len(predicted_high))
    figure,axis= plt.subplots(3)
    figure.set_size_inches(12.5, 12.5)
    axis[0].plot(X,predicted_high,color='black', linewidth=1, label='recovered_high')
    axis[0].plot(X,real_high,color='red', linewidth=1, label='real')
    axis[0].legend(loc='upper right', frameon=False, fontsize=15)
    axis[1].plot(X,predicted_low,color='black',linewidth=1,label='recovered_low')
    axis[1].plot(X,real_low,color='red', linewidth=1, label='real')
    axis[1].legend(loc='upper right', frameon=False, fontsize=15)
    axis[2].plot(X,predicted_close,color='black',linewidth=1,label='recovered_close')
    axis[2].plot(X,real_close, color='red',linewidth=1,label='real')
    axis[2].legend(loc='upper right', frameon=False, fontsize=15)
    plt.show()



if __name__ == '__main__':
    model = LinearAutoEncoder(in_features=5, hidden_size=[10,8], out_features=3)
    train(model)
    test_loader = test(model)
    torch.save(model.state_dict(), 'E:/baselines/DeepPocket-master/environment/aec_wrapper/autoencoder/linear_autoencoder_sp500.pt')
    plot(model,test_loader)




