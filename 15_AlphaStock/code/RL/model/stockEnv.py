import pandas as pd
import numpy as np
import torch
from config import Config

INITIAL_ACCOUNT_BALANCE = 1e5


class StockTradingEnv():
    """A stock trading environment for OpenAI gym"""
    def __init__(self, cfg):
        super(StockTradingEnv, self).__init__()
        self.train_barrier = '2022-12-26'
        self.test_barrier = '2022-12-26'  # 向前windows天

        tag = cfg.mode
        #self.df = pd.read_csv(cfg.pv_path, index_col =  "Unnamed: 0")
        self.df = pd.read_csv(cfg.pv_path)
        self.df['dt'] = pd.to_datetime(self.df['dt'])
        self.df = self.df.set_index('dt').sort_values(by=['dt','kdcode'])
        if tag == 'train':
            self.df = self.df.loc[:self.train_barrier]
        elif tag == 'test':
            self.df = self.df.loc[self.test_barrier:]
        else:
            raise NotImplementedError

        self.stock_num = self.df.kdcode.nunique()
        self.date_list = self.df.index.unique()
        print(self.date_list.shape, ' ', self.stock_num)
        self.featureList = cfg.featureList #['close','open','high','low','to','vol']
        self.window = 5


    def _next_observation(self):
        
        obs = self.df.loc[str(self.date_list[np.where(np.array(self.date_list) == self.day)[0][0] - (self.window-1)]):str(self.day)][self.featureList].values.reshape(self.stock_num,self.window,len(self.featureList))

        return obs

    def _take_action(self, action):
        '''
        action:[batch, stock_num, 3]
        0:持有 1:买入  2:卖出
        '''
        
        #price_list = self.df.loc[self.day].price.values
        price_list = self.df.loc[self.day].close.values
        # 不用挑选 直接全部卖出就行 持仓为0的利润自然就是0
        self.balance+=np.sum(self.shares_held*price_list)
        self.shares_held = np.zeros_like(self.shares_held)
        
        '''
        sell_idx_list = np.where(np.abs(self.shares_held) > 1e-5)[0]
        
        
        for sell_stock in sell_idx_list:
            if(self.shares_held[sell_stock]):
                self.balance+=self.shares_held[sell_stock]*price_list[sell_stock]
                self.shares_held[sell_stock] = 0
        '''

        buy_idx_list = action
        buy_num = len(buy_idx_list)
        money = self.balance/buy_num
        for buy_stock in buy_idx_list:
            share = int(money/price_list[buy_stock])
            self.shares_held[buy_stock] += share
            self.balance-=share*price_list[buy_stock]
            
        return

    def step(self, action):
        # Execute one time step within the environment
        self.day_idx += 1
        self.day = self.date_list[self.day_idx]
        self._take_action(action)


        price_list = self.df.loc[self.day].close.values
        reward = self.balance + (self.shares_held*price_list).sum() -self.last_value
        self.last_value = self.balance + (self.shares_held*price_list).sum()
        done = (self.day_idx == len(self.date_list) - 1)


        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        '''
        reset:
        '''
        self.balance = INITIAL_ACCOUNT_BALANCE          # cash
        self.shares_held = np.zeros(self.stock_num)     # stock_share
        self.last_value = INITIAL_ACCOUNT_BALANCE       # yesterday_value
        self.day_idx = self.window
        self.day = self.date_list[self.day_idx]          # day

        return self._next_observation()

    def render(self):
        # Render the environment to the screen
        profit = self.last_value - INITIAL_ACCOUNT_BALANCE

        print(f'Day: {self.day}')
        print(f'Balance: {self.balance}')
        print(f'profit: {profit}')
        print(f'Shares held: {self.shares_held})')


'''
cfg = Config()
env = StockTradingEnv(cfg)
obs, g_concept, g_industry, g_pos, g_neg= env.reset()
print(obs.shape, g_concept.shape, g_industry.shape, g_pos.shape, g_neg.shape)

action = [0,1,2]
env.step(action)
action = [2]
env.step(action)

env.render()
action = np.zeros(df.stockcode.unique().shape[0])+2
env.step(action)
'''