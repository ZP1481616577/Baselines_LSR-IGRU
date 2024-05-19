import random
import os
import numpy as np
import copy
import torch
import gym
from tqdm import tqdm

from stockEnv import StockTradingEnv
from config import plot_rewards,Config
from ppo import Agent

def train(cfg, env, agent):
    ''' 
    训练
    '''
    print("开始训练啦！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    best_epoch_reward = -1e7 # 记录回合最佳奖励
    output_agent = None
    for epoch in range(cfg.train_eps):
        # epoch训练-回合
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in tqdm(range(cfg.max_steps)):
            # 一个epoch 内进行训练
            ep_step += 1
            action, log_probs = agent.sample_action(state)  # 选择动作
            if len(action.shape) == 2:
                action = action.squeeze(0)
            if len(log_probs.shape) == 2:
                log_probs = log_probs.squeeze(0)
            
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            
            agent.memory.push((state, action, log_probs, reward, done))  # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            if done:
                break

        print(f"回合：{epoch+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}")

        if (epoch+1)%cfg.eval_per_episode == 0:
            # 遍历指定epoch数量后还是evaluate
            sum_eval_reward = 0
            eval_cfg = Config()
            eval_cfg.mode = 'test'
            eval_env = StockTradingEnv(eval_cfg)
            for _ in range(eval_cfg.eval_eps):
                # 因为动作选择有随机性 所以测试多个取平均值
                eval_ep_reward = 0
                state = eval_env.reset()  # 重置环境，返回初始状态
                for _ in range(eval_cfg.max_steps):
                    action = agent.predict_action(state)  # 选择动作
                    if len(action.shape) == 2:
                        action = action.squeeze(0)


                    next_state, reward, done, _ = eval_env.step(action)  # 更新环境，返回transition
                    
                    state = next_state  # 更新下一个状态
                    eval_ep_reward += reward  # 累加奖励
                    if done:
                        break
                sum_eval_reward += eval_ep_reward
                #eval_env.render()
            mean_eval_reward = sum_eval_reward/cfg.eval_eps
            if mean_eval_reward >= best_epoch_reward:
                best_epoch_reward = mean_eval_reward
                output_agent = copy.deepcopy(agent)
                print(f"回合：{epoch+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_epoch_reward:.2f}，更新模型！")
            else:
                print(f"回合：{epoch+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_epoch_reward:.2f}")
        steps.append(ep_step)
        rewards.append(ep_reward)
    print("完成训练！")
    return output_agent,{'rewards':rewards}

def test(cfg, env, agent):
    print("开始测试！")
    rewards = []  # 记录所有回合的奖励
    steps = []

    
    cumulative_returns = [[] for i in range(cfg.test_eps)]
    for epoch in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        cumulative_returns[epoch].append(env.last_value)

        for _ in range(cfg.max_steps):
            ep_step+=1
            action = agent.predict_action(state)  # 选择动作
            if len(action.shape) == 2:
                action = action.squeeze(0)
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            
            cumulative_returns[epoch].append(env.last_value)
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{epoch+1}/{cfg.test_eps}，奖励：{ep_reward:.2f}")
    print("完成测试")
    #env.close()

    return {'rewards':rewards, 'cumulative_returns':cumulative_returns[-1]}

def all_seed(seed = 1):
    ''' 万能的seed函数
    '''
    if seed == 0:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def env_agent_config(cfg):
    all_seed(seed=cfg.seed)

    # 创建环境
    env = StockTradingEnv(cfg)
    '''
    env = gym.make(cfg.env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    '''
    n_states = len(env.featureList)
    n_actions = env.stock_num


    print(f"状态空间维度: {n_states}, 动作空间维度: {n_actions}, epoch内交互次数: {len(env.date_list)}, window-len: {env.window}")
    # 更新n_states和n_actions到cfg参数中
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions) 
    setattr(cfg, 'max_steps', len(env.date_list))
    setattr(cfg, 'num_stocks', env.stock_num)
    setattr(cfg, 'window', env.window)
    agent = Agent(cfg)
    return env, agent

if __name__ == "__main__":
    # 获取参数
    cfg = Config() 
    # 训练
    cfg.mode = 'train'
    env, agent = env_agent_config(cfg)
    best_agent,res_dic = train(cfg, env, agent)
    plot_rewards(res_dic['rewards'], cfg, tag="train")  
    # 测试
    cfg.mode = 'test'
    test_env, _ = env_agent_config(cfg)
    torch.save(best_agent, cfg.save_path)
    res_dic = test(cfg, test_env, best_agent)
    plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果
    plot_rewards(res_dic['cumulative_returns'], cfg, tag="test", name='_CR')  # 画出结果

    np.save('../res/best_test_cr_whole.npy', np.array(res_dic['cumulative_returns']))
