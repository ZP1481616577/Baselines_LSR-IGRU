from agent.actor import Actor
from agent.critic import Critic
import torch
from agent.replay_memory import ReplayBuffer
from torch.distributions import Normal

class Agent():

    def __init__(self, in_channels,assets_number,gnn_parameters, trading_window_size,actor_lr,critic_lr,actor_weight_decay,critic_weight_decay,gamma,batch_size,mem_size,input_dims,sample_bias,nb):
        self.actor = Actor(in_channels,trading_window_size,actor_lr = actor_lr,gnn_parameters = gnn_parameters,actor_weight_decay = actor_weight_decay)
        self.critic = Critic(in_channels,assets_number,trading_window_size,lr = critic_lr,weight_decay = critic_weight_decay)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size,trading_window_size, input_dims,assets_number,batch_size,sample_bias)
        self.nb = nb

    def get_action(self,obs,prev_weigths):
        return self.actor(obs, prev_weigths)

    def store_transition(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def sample_memory(self):
        state, actions, reward, new_state = self.memory.sample_buffer()
        states = torch.tensor(state)
        actions = torch.stack(actions)
        rewards = torch.tensor(reward)
        states_ = torch.tensor(new_state)

        return states, actions, rewards, states_
    
    def learn(self):
        if self.memory.mem_cntr < self.memory.mem_size:
            return
        actor_loss = 0
        critic_loss = 0
        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        for _ in range(self.nb):
            states, actions, rewards, states_, = self.sample_memory()

            q_pred = self.critic(states)
            q_next = self.critic(states_)
            adv =  rewards + self.gamma* q_next - q_pred
            critic_loss += (adv *q_pred).mean()
            
            x = actions.clone().detach()

            mean,std  = torch.mean(x, dim=0), torch.std(x, dim = 0)

            mean = torch.clip(mean, min = 1e-6, max = 60)
            std = torch.clip(std,min = 1e-6,max = 30)
            dist = Normal(mean,std)

            actor_loss += (dist.log_prob(actions).sum() * adv.clone().detach()).mean()
            #actor_loss += (actions.log().sum()*adv.detach()).sum()

        actor_loss.backward()
        critic_loss.backward()
        self.critic.optimizer.step()
        self.actor.optimizer.step()
        self.memory.reset()
    
        
