
import torch

from config import plot_rewards,Config
from main import *


def backtest(cfg, env, agent):
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


if __name__ == "__main__":
    cfg = Config()  
    # 测试
    cfg.mode = 'test'
    cfg.dataset = 'CSI'
    test_env, _ = env_agent_config(cfg)
    test_agent = torch.load("../save_models/CSI-RL-All-0514-Model")
    res_dic = backtest(cfg, test_env, test_agent)
    # print(np.array(res_dic['weight_list_1']).shape, np.array(res_dic['weight_list_2']).shape)
    # graph_att_weights = np.array(res_dic['weight_list_1'][240:480]).squeeze(1)
    # hier_att_weights = np.array(res_dic['weight_list_2'][240:480]).squeeze(1)
    # np.save('/home/bianyuxuan/2023-CIKM/code/RL/res/graph_att_weights.npy', graph_att_weights)
    # np.save('/home/bianyuxuan/2023-CIKM/code/RL/res/hier_att_weights.npy', hier_att_weights)
    #plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果
    #plot_rewards(res_dic['cumulative_returns'], cfg, tag="test", name='_CR')  # 画出结果
    #np.save('/home/bianyuxuan/2023-CIKM/code/RL/res/NDX-PPO-0514-CR-Data.npy', np.array(res_dic['cumulative_returns'])) 
    #a = np.load('/home/bianyuxuan/2023-CIKM/code/RL/res/NDX-PPO-0514-CR-Data.npy')
    plot_rewards(res_dic['cumulative_returns'], cfg, tag="test", name='_backtest')  # 画出结果
