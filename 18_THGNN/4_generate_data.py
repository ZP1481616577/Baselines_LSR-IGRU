import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import pandas as pd
from torch.autograd import Variable


# feature_cols = ['open','high','low','close','volume']
feature_cols = ['open','high','low','close','volume','turnover']

stock_name = "hs300"
path1 = f"{stock_name}fealab.pkl"
df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')
relation = os.listdir(f'{stock_name}/relation/')
relation = sorted(relation)
date_unique=df1['dt'].unique()
stock_trade_data=date_unique.strftime('%Y-%m-%d').tolist()
stock_trade_data.sort()

df1['dt']=df1['dt'].astype('datetime64[ns]')


def fun1(relation_dt, start_dt_month, end_dt_month,df1):
    prev_date_num = 20
    adj_all = pd.read_csv(f'{stock_name}/relation/'+relation_dt+'.csv', index_col=0)
    adj_stock_set = list(adj_all.index)
    pos_g = nx.Graph(adj_all > 0.1)
    pos_adj = nx.adjacency_matrix(pos_g).toarray()
    pos_adj = pos_adj - np.diag(np.diag(pos_adj))
    pos_adj = torch.from_numpy(pos_adj).type(torch.float32)
    neg_g = nx.Graph(adj_all < -0.1)
    neg_adj = nx.adjacency_matrix(neg_g)
    neg_adj.data = np.ones(neg_adj.data.shape)
    neg_adj = neg_adj.toarray()
    neg_adj = neg_adj - np.diag(np.diag(neg_adj))
    neg_adj = torch.from_numpy(neg_adj).type(torch.float32)
    print('neg_adj over')
    print(neg_adj.shape)
    dts = stock_trade_data[stock_trade_data.index(start_dt_month):stock_trade_data.index(end_dt_month)+1]
    print(dts)
    for i in tqdm(range(len(dts))):
        end_data=dts[i]
        start_data = stock_trade_data[stock_trade_data.index(end_data)-(prev_date_num - 1)]
        df2 = df1[df1['dt'] <= end_data]
        df2 = df2[df2['dt'] >= start_data]
        kdcode = adj_stock_set
        feature_all = []
        mask = []
        labels = []
        day_last_kdcode = []
        for j in range(len(kdcode)):
            df3 = df2[df2['kdcode'] == kdcode[j]]
            y = df3[feature_cols].values
            if y.T.shape[1] == prev_date_num:
                one = []
                feature_all.append(y)
                mask.append(True)
                label = df3.loc[df3['dt'] == end_data]['label'].values
                labels.append(label[0])
                one.append(kdcode[j])
                one.append(end_data)
                day_last_kdcode.append(one)
        feature_all = np.array(feature_all)
        features = torch.from_numpy(feature_all).type(torch.float32)
        mask = [True]*len(labels)
        labels = torch.tensor(labels, dtype=torch.float32)
        result = {'pos_adj': Variable(pos_adj), 'neg_adj': Variable(neg_adj),  'features': Variable(features),
                  'labels': Variable(labels), 'mask': mask}
        with open(f'{stock_name}/data_train_predict/'+end_data+'.pkl', 'wb') as f:
            pickle.dump(result, f)
        df = pd.DataFrame(columns=['kdcode', 'dt'], data=day_last_kdcode)
        df.to_csv(f'{stock_name}/kdcode/'+end_data+'.csv', header=True, index=False, encoding='utf_8_sig')

# prev_date_num 用过去多少天的数据生成每天的数据
# 0.1 -0.1 相关性的参数

# 第一个参数为每个月的最后一个交易日
# 第二个参数为每个月的第一个交易日
# 第三个参数和第一个参数一样
'''fun1('2020-01-23', '2020-01-02', '2020-01-23', df1)
fun1('2020-02-28', '2020-02-03', '2020-02-28', df1)
fun1('2020-03-31', '2020-03-02', '2020-03-31', df1)
fun1('2020-04-30', '2020-04-01', '2020-04-30', df1)
fun1('2020-05-29', '2020-05-06', '2020-05-29', df1)
fun1('2020-06-30', '2020-06-01', '2020-06-30', df1)
fun1('2020-07-31', '2020-07-01', '2020-07-31', df1)
fun1('2020-08-31', '2020-08-03', '2020-08-31', df1)
fun1('2020-09-30', '2020-09-01', '2020-09-30', df1)
fun1('2020-10-30', '2020-10-09', '2020-10-30', df1)
fun1('2020-11-30', '2020-11-02', '2020-11-30', df1)
fun1('2020-12-30', '2020-12-01', '2020-12-30', df1)'''

'''for i in ['2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']:
    for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        stock_m=[k for k in stock_trade_data if k>i+'-'+j and k<i+'-'+j+'-32']
        fun1(stock_m[-1], stock_m[0], stock_m[-1], df1)
for j in ['04','05','06','07','08','09','10','11','12']:
    stock_m=[k for k in stock_trade_data if k>'2008'+'-'+j and k<'2008'+'-'+j+'-32']
    fun1(stock_m[-1], stock_m[0], stock_m[-1], df1)'''

for i in ['2018','2019','2020','2021','2022','2023']:
    for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        stock_m=[k for k in stock_trade_data if k>i+'-'+j and k<i+'-'+j+'-32']
        fun1(stock_m[-1], stock_m[0], stock_m[-1], df1)

