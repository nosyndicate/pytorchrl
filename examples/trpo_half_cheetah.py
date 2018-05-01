import numpy as np
import gym

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from pytorchrl.algos.trpo import TRPO
from pytorchrl.policies.gaussian_mlp_policy import GaussianMLPPolicy
from pytorchrl.misc.log_utils import logdir

def main():
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

if __name__ == "__main__":
    main()
