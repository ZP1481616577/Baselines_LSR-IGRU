from net import ActorSoftmax, Critic
from replaybuffer import PGReplay
from torch.distributions import Categorical
import torch
import numpy as np
class Agent:
    def __init__(self,cfg) -> None:
        
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device) 
        self.actor = ActorSoftmax(cfg.n_states, hidden_dim = cfg.actor_hidden_dim,n_encoder_layers=cfg.n_encoder_layers, n_heads=cfg.n_heads, negative_slope=cfg.negative_slope,num_stocks=cfg.num_stocks).to(self.device)
        self.critic = Critic(input_dim=cfg.n_states, output_dim=1, hidden_dim=cfg.critic_hidden_dim, n_encoder_layers=cfg.n_encoder_layers, n_heads=cfg.n_heads, negative_slope=cfg.negative_slope,num_stocks=cfg.num_stocks).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PGReplay()
        self.k_epochs = cfg.k_epochs # update policy for K epochs
        self.eps_clip = cfg.eps_clip # clip parameter for PPO
        self.entropy_coef = cfg.entropy_coef # entropy coefficient
        self.sample_count = 0
        self.update_freq = cfg.update_freq

        self.window = cfg.window
        self.num_stocks = int(cfg.num_stocks/10)

    def sample_action(self,state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        
        '''

        g_concept = torch.tensor(g_concept, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        g_industry = torch.tensor(g_industry, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        g_pos = torch.tensor(g_pos, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        g_neg = torch.tensor(g_neg, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        '''

        
        probs = self.actor(state)

        dist = Categorical(probs)
        '''
        action = dist.sample()
        log_probs = dist.log_prob(action).detach()
        '''
        action = torch.multinomial(probs, self.num_stocks)
        log_probs = torch.tensor([0.], device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        for idx in range(self.num_stocks):
            log_probs += dist.log_prob(action[0][idx]).detach()
        log_probs/=self.num_stocks

        
        return action.detach().cpu().numpy(), log_probs
    @torch.no_grad()
    def predict_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        '''
        g_concept = torch.tensor(g_concept, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        g_industry = torch.tensor(g_industry, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        g_pos = torch.tensor(g_pos, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        g_neg = torch.tensor(g_neg, device=self.device, dtype=torch.float32).unsqueeze(dim=0)

        '''
        

        probs = self.actor(state)

        '''
        action = dist.sample()
        dist = Categorical(probs)
        '''
        action = torch.multinomial(probs, self.num_stocks)
        return action.detach().cpu().numpy()
    
    def update(self):
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()

        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)

        '''
        old_g_concept = torch.tensor(np.array(old_g_concept), device=self.device, dtype=torch.float32)
        old_g_industry = torch.tensor(np.array(old_g_industry), device=self.device, dtype=torch.float32)
        old_g_pos = torch.tensor(np.array(old_g_pos), device=self.device, dtype=torch.float32)
        old_g_neg = torch.tensor(np.array(old_g_neg), device=self.device, dtype=torch.float32)
        '''
        
        

        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        '''
        torch.Size([100, 255, 20, 6]) torch.Size([100, 255, 255]) torch.Size([100, 255, 255]) torch.Size([100, 25]) torch.Size([100])
        print(old_states.shape, old_g_concept.shape, old_g_neg.shape, old_actions.shape, old_log_probs.shape,)
        '''
        
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        for _ in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states) # detach to avoid backprop through the critic
            # returns[100] values[100,1]
            returns = returns.reshape(values.shape)
            advantage = returns - values.detach()
            # get action probabilities
            probs = self.actor(old_states)
            dist = Categorical(probs)
            # get new action probabilities
            '''
            new_probs = dist.log_prob(old_actions)
            '''
            new_probs = torch.zeros(old_log_probs.shape, device=self.device, dtype=torch.float32)
            for batch in range(old_log_probs.shape[0]):
                for idx in range(self.num_stocks):
                    new_probs[batch] += dist.log_prob(old_actions[batch][idx]).detach()[batch]
                new_probs[batch]/=self.num_stocks


            # compute ratio (pi_theta / pi_theta__old):
            # old_log_probs[100] new_probs[100,1]
            old_log_probs = old_log_probs.reshape(new_probs.shape)
            ratio = torch.exp(new_probs - old_log_probs) # old_log_probs must be detached
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # compute actor loss
            actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()