import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from psac.utils import soft_update, hard_update
from psac.model import GaussianPolicy, QNetwork, ValueNetwork


class SAC(object):
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy,target_update_interval,automatic_entropy_tuning,hidden_size,lr):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_range = [action_space.low, action_space.high]

        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        # critic is teh Q
        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        # value is the value entwrok
        self.value = ValueNetwork(num_inputs, hidden_size).to(device=self.device)
        self.value_target = ValueNetwork(num_inputs, hidden_size).to(self.device)
        self.value_optim = Adam(self.value.parameters(), lr=lr)
        hard_update(self.value_target, self.value)
        # the actor is the policy
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)



    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()[0]
        return self.rescale_action(action)
    
    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
                (self.action_range[1] + self.action_range[0]) / 2.0


    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            # this is the target value so we need to insert the next state
            vf_next_target = self.value_target(next_state_batch)
            # this is the next q value
            next_q_value = reward_batch + mask_batch * self.gamma * (vf_next_target)
        # we use two q functions
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, mean, log_std = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # Regularization Loss
        reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        policy_loss += reg_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        vf = self.value(state_batch)
        
        with torch.no_grad():
            vf_target = min_qf_pi - (self.alpha * log_pi)

        vf_loss = F.mse_loss(vf, vf_target) # JV = 𝔼(st)~D[0.5(V(st) - (𝔼at~π[Q(st,at) - α * logπ(at|st)]))^2]

        self.value_optim.zero_grad()
        vf_loss.backward()
        self.value_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.value_target, self.value, self.tau)

        return vf_loss.item(), qf1_loss.item(), qf2_loss.item(), policy_loss.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, value_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        if value_path is None:
            value_path = "models/sac_value_{}_{}".format(env_name, suffix)
        print('Saving models to {}, {} and {}'.format(actor_path, critic_path, value_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.value.state_dict(), value_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path, value_path):
        print('Loading models from {}, {} and {}'.format(actor_path, critic_path, value_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if value_path is not None:
            self.value.load_state_dict(torch.load(value_path))

