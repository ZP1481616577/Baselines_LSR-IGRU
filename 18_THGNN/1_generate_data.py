import os
os.environ["kdconfig"] = "config/database-dev.yaml"
os.environ["dbsync_config"] = "config/sync_config_dev.ini"
os.environ["task_config"] = "config/task_conf.ini"
from qsdata.api import *

kdcodes = None
start_dt = '2018-01-01'
end_dt = '2022-06-30'

# 获取原始数据
fields = ['close', 'open', 'high', 'low', 'volume', 'turnover', 'prev_close', 'adjfactor']
df_market = get_price(kdcodes, start_dt, end_dt, fields=fields, adjust_type='none')
df_market = df_market.reset_index()
df_market.to_pickle("data_2018_2022.pkl")

# 获取标签数据
df_vwap = get_vwap(kdcodes, start_date=start_dt, end_date=end_dt, adjust_type='post')
df_vwap_sorted = df_vwap.reset_index().sort_values(['kdcode', 'dt'])
df_vwap_sorted.to_pickle("data_2018_2022_label.pkl")
