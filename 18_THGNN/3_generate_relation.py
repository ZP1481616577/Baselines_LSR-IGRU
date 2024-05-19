import os
import time
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# 使用的特征列名称
# feature_cols = ['high','low','close','open','volume']
feature_cols = ['high','low','close','open','volume', 'turnover']
# 所有的交易日列表

def cal_pccs(x, y, n):
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    # print((n*sum_x2-sum_x*sum_x), ' ', (n*sum_y2-sum_y*sum_y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

def calculate_pccs(xs, yss, n):
    result = []
    for name in yss:
        ys = yss[name]
        tmp_res = []
        for pos, x in enumerate(xs):
            y = ys[pos]
            tmp_res.append(cal_pccs(x, y, n))
        result.append(tmp_res)
    return np.mean(result, axis=1)

def stock_cor_matrix(ref_dict, kdcodes, n, processes=1):
    if processes > 1:
        pool = mp.Pool(processes=processes)
        args_all = [(ref_dict[kdcode], ref_dict, n) for kdcode in kdcodes]
        results = [pool.apply_async(calculate_pccs, args=args) for args in args_all]
        output = [o.get() for o in results]
        data = np.stack(output)
        return pd.DataFrame(data=data, index=kdcodes, columns=kdcodes)
    data = np.zeros([len(kdcodes), len(kdcodes)])
    for i in range(len(kdcodes)):
        data[i, :] = calculate_pccs(ref_dict[kdcodes[i]], ref_dict, n)
    return pd.DataFrame(data=data, index=kdcodes, columns=kdcodes)


stock_name = "zz500"
path1 = f"{stock_name}fealab.pkl"
df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')
# df1['dt']=df1['dt'].astype('datetime64')
prev_date_num = 20
date_unique=df1['dt'].unique()
stock_trade_data=date_unique.strftime('%Y-%m-%d').tolist()
stock_trade_data.sort()


# dt为每一个的最后一个交易日
'''dt = ['2020-01-23', '2020-02-28', '2020-03-31', '2020-04-30', '2020-05-29', '2020-06-30',
       '2020-07-31', '2020-08-31', '2020-09-30', '2020-10-30', '2020-11-30', '2020-12-31',
       '2021-01-29', '2021-02-26', '2021-03-31', '2021-04-30', '2021-05-31', '2021-06-30',
       '2021-07-30', '2021-08-31', '2021-09-30', '2021-10-29', '2021-11-30', '2021-12-31',
       '2022-01-28', '2022-02-28', '2022-03-31', '2022-04-29', '2022-05-31']'''
'''dt = ['','2020-01-23', '2020-02-28', '2020-03-31', '2020-04-30', '2020-05-29', '2020-06-30',
       '2020-07-31', '2020-08-31', '2020-09-30', '2020-10-30', '2020-11-30', '2020-12-30',
       '2021-01-29', '2021-02-26', '2021-03-31', '2021-04-30', '2021-05-31', '2021-06-30',
       '2021-07-30', '2021-08-31', '2021-09-30', '2021-10-29', '2021-11-30','2021-12-29']'''
dt=[]

for i in ['2018','2019','2020','2021','2022','2023']:
    for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        stock_m=[k for k in stock_trade_data if k>i+'-'+j and k<i+'-'+j+'-32']
        dt.append(stock_m[-1])

df1['dt']=df1['dt'].astype('datetime64[ns]')


for i in range(len(dt)):
    print(i)
    df2 = df1.copy()
    end_data = dt[i]
    start_data = stock_trade_data[stock_trade_data.index(end_data)-(prev_date_num - 1)]
    df2 = df2.loc[df2['dt'] <= end_data]
    df2 = df2.loc[df2['dt'] >= start_data]
    kdcode = sorted(list(set(df2['kdcode'].values.tolist())))
    test_tmp = {}
    for j in tqdm(range(len(kdcode))):
        df3 = df2.loc[df2['kdcode'] == kdcode[j]]
        y = df3[feature_cols].values
        if y.T.shape[1] == prev_date_num:
            test_tmp[kdcode[j]] = y.T
    t1 = time.time()
    result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num, processes=1)
    result=result.fillna(0)
    for i in range(len(kdcode)):
        result.iloc[i,i]=1
    t2 = time.time()
    print('time cost', t2 - t1, 's')
    result.to_csv(f"{stock_name}/relation/"+str(end_data)+".csv")

# prev_date_num 用过去多少天的数据计算相似度，
# np.mean(result, axis=1) 现在是求6列因子相似度的均值，可以考虑求中位数