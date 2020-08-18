import math

import numpy as np
import torch
from cpprb import ReplayBuffer
from scipy.signal import lfilter


def calculate_log_pi(log_stds, noises, actions):
    return (
            (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True)
            - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
            - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True))


def reparameterize(means, log_stds):
    noises = torch.randn_like(means)

    actions = means + noises * log_stds.exp()
    actions = torch.tanh(actions)

    log_pis = calculate_log_pi(log_stds, noises, actions)
    return actions, log_pis


def atanh(x):
    # inverse hyperbolic tangent
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def compute_log_probs(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_replay_buffer(policy, env, episode_max_steps):
    rb_dict = {
        "size": policy.horizon,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {"shape": env.observation_space.shape},
            "act": {"shape": env.action_space.shape},
            "done": {},
            "logp": {},
            "ret": {},
            "adv": {}}}
    on_policy_buffer = ReplayBuffer(**rb_dict)

    rb_dict = {
        "size": episode_max_steps,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {"shape": env.observation_space.shape},
            "act": {"shape": env.action_space.shape},
            "next_obs": {"shape": env.observation_space.shape},
            "rew": {},
            "done": {},
            "logp": {},
            "val": {}}}
    episode_buffer = ReplayBuffer(**rb_dict)

    return on_policy_buffer, episode_buffer


def discount_cumsum(x, discount):
    """
    Forked from rllab for computing discounted cumulative sums of vectors.

    :param x (np.ndarray or tf.Tensor)
        vector of [x0, x1, x2]
    :return output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return lfilter(
        b=[1],
        a=[1, float(-discount)],
        x=x[::-1],
        axis=0)[::-1]
