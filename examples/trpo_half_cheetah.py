import numpy as np
import gym

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from pytorchrl.algos.trpo import TRPO
from pytorchrl.envs.gym_env import GymEnv
from pytorchrl.policies.gaussian_mlp_policy import GaussianMLPPolicy
from pytorchrl.misc.log_utils import logdir
from pytorchrl.misc.instrument import run_experiment_lite, VariantGenerator, variant

class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1]

def run_task(v):
    env = GymEnv('HalfCheetah-v1', record_video=False, force_reset=True)
    observation_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)
    policy = GaussianMLPPolicy(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_sizes=(400, 300))

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        n_itr=1000,
        batch_size=50000,
        max_path_length=500,
        discount=0.99,
        step_size=0.01,
        use_finite_diff_hvp=True,
        # plot=True,
    )

    algo.train()

variants = VG().variants()

for v in variants:
    run_experiment_lite(
        run_task,
        exp_prefix='trpo_half_cheetah',
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode='last',
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v['seed'],
        variant=v,
        mode='local',
        # dry=True,
        # plot=True,
        # terminate_machine=False,
    )

