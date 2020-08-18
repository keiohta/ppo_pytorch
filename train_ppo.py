import gym
import numpy as np

from ppo import PPO
from utils import get_replay_buffer, discount_cumsum


def main():
    # Env
    env = gym.make("Pendulum-v0")
    episode_max_steps = 200
    test_episodes = 10

    # Policy
    policy = PPO(env.observation_space.shape, env.action_space.shape, max_action=env.action_space.high[0])

    on_policy_buffer, episode_buffer = get_replay_buffer(policy, env, episode_max_steps)

    def collect_transitions():
        on_policy_buffer.clear()
        n_episodes = 0
        ave_episode_return = 0
        while on_policy_buffer.get_stored_size() < policy.horizon:
            obs = env.reset()
            episode_return = 0.
            for i in range(episode_max_steps):
                act, logp, val = policy.get_action_and_val(obs)
                next_obs, rew, done, _ = env.step(act)
                episode_buffer.add(obs=obs, act=act, next_obs=next_obs, rew=rew,
                                   done=done, logp=logp, val=val)
                obs = next_obs
                if done:
                    break
                episode_return += rew
            finish_horizon(last_val=val)
            ave_episode_return += episode_return
            n_episodes += 1
        finish_horizon(last_val=val)

        return ave_episode_return / n_episodes

    def finish_horizon(last_val=0):
        samples = episode_buffer.get_all_transitions()
        rews = np.append(samples["rew"], last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda
        deltas = rews[:-1] + policy.discount * vals[1:] - vals[:-1]
        advs = discount_cumsum(deltas, policy.discount * policy.lam)

        # Compute targets for value function
        rets = discount_cumsum(rews, policy.discount)[:-1]
        on_policy_buffer.add(
            obs=samples["obs"], act=samples["act"], done=samples["done"],
            ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))
        episode_buffer.clear()

    def update_policy():
        # Compute means and variances
        samples = on_policy_buffer.get_all_transitions()
        mean_adv = np.mean(samples["adv"])
        std_adv = np.std(samples["adv"])

        for _ in range(policy.n_epoch):
            samples = on_policy_buffer._encode_sample(np.random.permutation(policy.horizon))
            adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)
            for idx in range(int(policy.horizon / policy.batch_size)):
                target = slice(idx * policy.batch_size, (idx + 1) * policy.batch_size)
                policy.train(
                    states=samples["obs"][target],
                    actions=samples["act"][target],
                    advantages=adv[target],
                    logp_olds=samples["logp"][target],
                    returns=samples["ret"][target])

    def eval_policy(visualize=False):
        avg_test_return = 0.
        for i in range(test_episodes):
            episode_return = 0.
            obs = env.reset()
            if visualize:
                env.render()
            for _ in range(episode_max_steps):
                act = policy.get_action(obs, test=True)
                next_obs, rew, _, _ = env.step(act)
                if visualize:
                    env.render()
                episode_return += rew
                obs = next_obs
            avg_test_return += episode_return
        return avg_test_return / test_episodes

    n_updates = 0
    while True:
        n_updates += 1

        collect_transitions()
        update_policy()
        if n_updates % 10 == 0:
            ave_return = eval_policy()
            print("n_update: {: 3d} return: {: .4f}".format(n_updates, ave_return))

        if n_updates % 50 == 0:
            eval_policy(visualize=True)


if __name__ == "__main__":
    main()
