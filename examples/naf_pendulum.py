import numpy as np
import gym


from rllab.envs.normalized_env import normalize

from pytorchrl.algos.naf import NAF
from pytorchrl.envs.gym_env import GymEnv
from pytorchrl.exploration_strategies.ou_strategy import OUStrategy
from pytorchrl.q_functions.normalized_adv_function import NormalizedAdvantageFunction
from pytorchrl.misc.instrument import run_experiment_lite, VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1, 11, 21, 31, 41]

    @variant
    def name(self):
        return ['pytorch']


def run_task(*_):
    env = normalize(GymEnv("Pendulum-v0", record_video=False, force_reset=True))

    observation_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)

    qf = NormalizedAdvantageFunction(
        observation_dim=observation_dim,
        action_dim=action_dim,
        vf_hidden_sizes=(200, 200),
        mean_hidden_sizes=(200, 200),
        pds_hidden_sizes=(200, 200))

    es = OUStrategy(action_space=env.action_space)

    algo = NAF(
        env=env,
        es=es,
        qf=qf,
        batch_size=64,
        max_path_length=100,
        epoch_length=1000,
        min_pool_size=1000,
        n_epochs=80,
        discount=0.99,
        # scale_reward=0.01,
        qf_learning_rate=1e-3,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        plot=True,
    )
    algo.train()

# if __name__ == '__main__':
#     run_task()

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

