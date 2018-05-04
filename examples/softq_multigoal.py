import numpy as np
import gym

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from pytorchrl.algos.softq import SoftQ
from pytorchrl.q_functions.svgd_mlp_q_function import SVGDMLPQFunction
from pytorchrl.policies.svgd_policy import SVGDPolicy
from pytorchrl.envs.multigoal_env import MultiGoalEnv
from pytorchrl.misc.instrument import run_experiment_lite, VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1]

def run_task(*_):
    env = normalize(MultiGoalEnv())

    observation_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)


    qf =  SVGDMLPQFunction(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_sizes=(100, 100),
    )

    policy = SVGDPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_sizes=(100, 100),
        output_nonlinearity=None,
    )

    algo = SoftQ(
        env=env,
        policy=policy,
        qf=qf,
        batch_size=64,
        n_epochs=100,
        epoch_length=100,
        min_pool_size=100,
        replay_pool_size=1000000,
        discount=0.99,
        alpha=0.1,
        max_path_length=30,
        qf_target_n_particles=16,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-3,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        n_eval_episodes=10,
        soft_target_tau=1000,
        scale_reward=0.1,
        include_horizon_terminal_transitions=False,
        # plot=True,
    )

    algo.train()

# if __name__ == '__main__':
#     run_task()

variants = VG().variants()

for v in variants:
    run_experiment_lite(
        run_task,
        exp_prefix="softq_multigoal",
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





