import numpy as np
import gym

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from pytorchrl.algos.trpo import TRPO
from pytorchrl.envs.gym_env import GymEnv
from pytorchrl.policies.gaussian_mlp_policy import GaussianMLPPolicy
from pytorchrl.misc.log_utils import logdir

def main():
    env = GymEnv('Pendulum-v0', record_video=False, force_reset=True)

    observation_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)

    policy = GaussianMLPPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_sizes=(100, 100))

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        batch_size=10000,
        max_path_length=env.horizon,
        n_itr=100,
        discount=0.99,
        store_paths=True,
        step_size=0.01,
    )

    with logdir(algo=algo, dirname='data/irl/pendulum'):
        algo.train()


if __name__ == '__main__':
    main()