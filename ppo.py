import torch
from torch import nn

from utils import fix_seed, reparameterize, compute_log_probs


class GaussianActor(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0]),
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return compute_log_probs(self.net(states), self.log_stds, actions)


class Critic(nn.Module):
    def __init__(self, state_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, states):
        return self.net(states)


class PPO:
    def __init__(self,
                 state_shape,
                 action_shape,
                 max_action=1.,
                 device=torch.device('cpu'),
                 seed=0,
                 batch_size=64,
                 lr=3e-4,
                 discount=0.9,
                 horizon=2048,
                 n_epoch=10,
                 clip_eps=0.2,
                 lam=0.95,
                 coef_ent=0.,
                 max_grad_norm=10.):
        fix_seed(seed)

        self.actor = GaussianActor(state_shape, action_shape).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_shape).to(device)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.device = device
        self.batch_size = batch_size
        self.discount = discount
        self.horizon = horizon
        self.n_epoch = n_epoch
        self.clip_eps = clip_eps
        self.lam = lam
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def get_action(self, state, test=False):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            if test:
                action = self.actor(state)
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0] * self.max_action

    def get_action_and_val(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, logp = self.actor.sample(state)
            value = self.critic(state)
        return action * self.max_action, logp, value

    def train(self, states, actions, advantages, logp_olds, returns):
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions / self.max_action).float()
        advantages = torch.from_numpy(advantages).float()
        logp_olds = torch.from_numpy(logp_olds).float()
        returns = torch.from_numpy(returns).float()
        self.update_actor(states, actions, logp_olds, advantages)
        self.update_critic(states, returns)

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, logp_olds, advantages):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mean_entropy = -log_pis.mean()

        ratios = (log_pis - logp_olds).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean() - self.coef_ent * mean_entropy

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()


if __name__ == "__main__":
    import gym
    env = gym.make("Pendulum-v0")

    policy = PPO(env.observation_space.shape, env.action_space.shape)

    obs = env.reset()
    act = policy.get_action(obs)
    print(act, type(act))
    act = policy.get_action(obs, test=True)
    print(act, type(act))
