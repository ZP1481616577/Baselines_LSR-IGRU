import matplotlib.pyplot as plt
import seaborn as sns


def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed
def plot_rewards(rewards,cfg, tag='train', name = ''):
    ''' 画图
    '''
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.savefig("../res/"+f"{tag}ing-curve-on-{cfg.device}-of-{cfg.algo_name}-for-{cfg.env_name}"+ name + ".jpg")

class Config:
    def __init__(self) -> None:
        '''
        1. 数据源部分
        '''
        self.dataset = 'CCSO'
        if self.dataset == 'HS300' or self.dataset == 'ZZ500':
            self.pv_path = f"E:/baselines/{self.dataset}.csv"
            self.featureList = ['close', 'open', 'high', 'low', 'turnover', 'volume']
        elif self.dataset == 'SP500' or self.dataset == 'NASDAQ100':
            self.pv_path = f"E:/baselines/{self.dataset}_new.csv"
            self.featureList = ['close', 'open', 'high', 'low', 'volume']
        elif self.dataset == 'CCSO':
            self.pv_path = f"E:/baselines/day_level_dataset/ccso_2022-09-01_2023-03-31.csv"
            self.featureList = ['close', 'Last', 'Volume', 'TurnOver', 'OpenPrice',
       'HighPrice', 'LowPrice', 'TotalNo', 'TotalSellOrderVolume',
       'TotalBuyOrderVolume', 'Ask1', 'AskVol1', 'Ask2', 'AskVol2', 'Ask3',
       'AskVol3', 'Ask4', 'AskVol4', 'Ask5', 'AskVol5', 'Ask6', 'AskVol6',
       'Ask7', 'AskVol7', 'Ask8', 'AskVol8', 'Ask9', 'AskVol9', 'Ask10',
       'AskVol10', 'Bid1', 'BidVol1', 'Bid2', 'BidVol2', 'Bid3', 'BidVol3',
       'Bid4', 'BidVol4', 'Bid5', 'BidVol5', 'Bid6', 'BidVol6', 'Bid7',
       'BidVol7', 'Bid8', 'BidVol8', 'Bid9', 'BidVol9', 'Bid10', 'BidVol10',
       'label']
        '''
        2. 训练与测试
        '''
        self.env_name = "StockTradingEnv" # 环境名字
        self.new_step_api = False # 是否用gym的新api
        self.algo_name = "PPO" # 算法名字
        self.mode = "train" # train or test
        self.seed = 42 # 随机种子
        self.device = "cuda" # device to use
        self.train_eps = 200 # 训练的回合数
        self.test_eps = 1 # 测试的回合数
        self.max_steps = 200 # 每个回合的最大步数
        self.eval_eps = 200 # 评估的回合数
        self.eval_per_episode = 10 # 评估的频率
        '''
        3. 网络参数
        '''
        self.gamma = 0.99 # 折扣因子
        self.k_epochs = 4 # 更新策略网络的次数 默认为4
        self.actor_lr = 0.00002 # actor网络的学习率
        self.critic_lr = 0.00002 # critic网络的学习率
        self.eps_clip = 0.2 # epsilon-clip
        self.entropy_coef = 0.01 # entropy的系数
        self.update_freq = 2 # 更新频率
        self.actor_hidden_dim = 32 # actor网络的隐藏层维度 32 64
        self.critic_hidden_dim = 32 # critic网络的隐藏层维度 32 64
        self.n_encoder_layers = 1 # 时序encoder中的隐藏叠加层数
        self.n_heads = 2 # 2 多头注意力机制头数
        self.negative_slope = 0.1 # leakyrelu


        self.save_path = "../save_models/model"