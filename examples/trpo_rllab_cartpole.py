from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite, VariantGenerator, variant
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]

    @variant
    def name(self):
        return ['pytorch-finite']

def run_task(*_):
    # Please note that different environments with different action spaces may
    # require different policies. For example with a Discrete action space, a
    # CategoricalMLPPolicy works, but for a Box action space may need to use
    # a GaussianMLPPolicy (see the trpo_gym_pendulum.py example)
    env = normalize(GymEnv("CartPole-v0", record_video=False, force_reset=True))

    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
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
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5, symmetric=False))
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()

variants = VG().variants()

for v in variants:
    run_experiment_lite(
        run_task,
        exp_prefix='trpo_cartpole_comparison',
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v['seed'],
        variant=v,
        # plot=True,
    )
