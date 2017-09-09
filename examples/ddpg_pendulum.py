import numpy as np
import gym

from pytorchrl.algos.ddpg import DDPG
from pytorchrl.exploration_strategies.ou_strategy import OUStrategy
from pytorchrl.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from pytorchrl.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction


def run_task(*_):
    env = gym.make('Pendulum-v0')

    observation_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)

    policy = DeterministicMLPPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(400, 300)
    )

    es = OUStrategy(action_space=env.action_space)

    qf = ContinuousMLPQFunction(observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_sizes=(400, 300))

    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=32,
        max_path_length=100,
        epoch_length=1000,
        min_pool_size=10000,
        n_epochs=1000,
        discount=0.99,
        scale_reward=0.01,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        plot=True,
    )
    algo.train()


if __name__ == '__main__':
    run_task()
