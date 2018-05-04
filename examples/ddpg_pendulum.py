import numpy as np
import gym

from pytorchrl.algos.ddpg import DDPG
from pytorchrl.envs.gym_env import GymEnv
from pytorchrl.envs.normalized_env import normalize
from pytorchrl.exploration_strategies.ou_strategy import OUStrategy
from pytorchrl.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from pytorchrl.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from pytorchrl.misc.instrument import run_experiment_lite, VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1]


def run_task(*_):
    env = normalize(GymEnv("Pendulum-v0", record_video=False, force_reset=True))

    observation_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)

    policy = DeterministicMLPPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(400, 300))

    es = OUStrategy(action_space=env.action_space)

    qf = ContinuousMLPQFunction(observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_sizes=(400, 300))

    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=64,
        max_path_length=100,
        epoch_length=1000,
        min_pool_size=1000,
        n_epochs=11,
        discount=0.99,
        scale_reward=0.01,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()

variants = VG().variants()

for v in variants:
    run_experiment_lite(
        run_task,
        exp_prefix="ddpg_pendulum_test",
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )

