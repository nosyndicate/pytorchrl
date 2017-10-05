import numpy as np
import gym

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator, variant

from pytorchrl.algos.trpo import TRPO
from pytorchrl.policies.categorical_mlp_policy import CategoricalMLPPolicy


def run_task(*_):
    env = normalize(GymEnv("CartPole-v0", record_video=False, force_reset=True))

    observation_dim = np.prod(env.observation_space.shape)
    num_actions = env.action_space.n

    policy = CategoricalMLPPolicy(
        observation_dim=observation_dim,
        num_actions=num_actions,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()

if __name__ == '__main__':
    run_task()

# run_experiment_lite(
#     run_task,
#     # Number of parallel workers for sampling
#     n_parallel=1,
#     # Only keep the snapshot parameters for the last iteration
#     snapshot_mode="last",
#     # Specifies the seed for the experiment. If this is not provided, a random seed
#     # will be used
#     seed=1,
#     # plot=True,
# )
