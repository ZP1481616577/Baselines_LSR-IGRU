import pickle
import pandas as pd

# 处理训练数据
path1 = "data_2018_2022.pkl"
df_market = pickle.load(open(path1, 'rb'), encoding='utf-8')
df_market['prev_adjfactor'] = df_market.groupby('kdcode')['adjfactor'].shift()
df_market = df_market[~df_market['prev_adjfactor'].isna()]
df_market['open_r'] = (df_market['open'] * df_market['adjfactor']) / (df_market['prev_close'] * df_market['prev_adjfactor']) #- 1
df_market['high_r'] = (df_market['high'] * df_market['adjfactor']) / (df_market['prev_close'] * df_market['prev_adjfactor']) #- 1
df_market['low_r'] = (df_market['low'] * df_market['adjfactor']) / (df_market['prev_close'] * df_market['prev_adjfactor']) #- 1
df_market['close_r'] = (df_market['close'] * df_market['adjfactor']) / (df_market['prev_close'] * df_market['prev_adjfactor']) #- 1
df_market['turnover_r'] = df_market['turnover'] / 10000000000
df_market["volume_r"] = df_market.groupby('kdcode')['volume'].pct_change().fillna(0) + 1
df_market_1 = df_market
df_market_1 = df_market_1.drop(columns=['close', 'open', 'high', 'low', 'volume', 'turnover', 'prev_close', 'adjfactor', 'prev_adjfactor'])
df_market_1 = df_market_1.rename(columns={'open_r': 'open','high_r': 'high','low_r': 'low','close_r': 'close','turnover_r': 'turnover','volume_r': 'volume'})

# 处理标签
path2 = "data_2018_2022_label.pkl"
df_vwap_sorted = pickle.load(open(path2, 'rb'), encoding='utf-8')
# 使用n天后的数据计算收益率
n = 5
# 使用什么价格成交
c = 'am-15m'
df_vwap_sorted['t1_{}'.format(c)] = df_vwap_sorted.groupby('kdcode')[c].shift(-1)
df_vwap_sorted['t{}_{}'.format(n, c)] = df_vwap_sorted.groupby('kdcode')[c].shift(-n)
df_vwap_sorted['t{}_{}_return_rate'.format(n, c)] = \
    (df_vwap_sorted['t{}_{}'.format(n, c)]) / (df_vwap_sorted['t1_{}'.format(c)]) - 1
df_vwap_sorted['dt'] = pd.to_datetime(df_vwap_sorted['dt'])


# 数据合并与处理
df_labeled_features = df_market_1.merge(df_vwap_sorted, how='inner', left_on=['kdcode', 'dt'],right_on=['kdcode', 'dt'])
df_labeled_features = df_labeled_features.dropna()
df_labeled_features = df_labeled_features.drop(columns = ['am-5m', 'am-15m', 'am-30m', 'am-60m', 'am-120m', 'pm-5m', 'pm-15m','pm-30m', 'pm-60m', 'pm-120m', 'pm-245m-255m', 't1_am-15m', 't5_am-15m'])
df_labeled_features = df_labeled_features.rename(columns={'t5_am-15m_return_rate':'t5_label'})
# print(df_labeled_features.columns.tolist())
# ['kdcode', 'dt', 'open', 'high', 'low', 'close', 'turnover', 'volume', 't5_label']
df_labeled_features.to_pickle("process_2018_2022_data_label.pkl")