import numpy as np
import numpy.random as nr

import gym


class OUStrategy(object):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process

    To understand the effect of each parameters, see
    http://www.math.ku.dk/~susanne/StatDiff/Overheads1b
    """

    def __init__(self, action_space, mu=0, theta=0.15, sigma=0.3, **kwargs):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = action_space
        self.action_dim = np.prod(self.action_space.shape)
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        ou_state = self.evolve_state()
        return np.clip(action + ou_state, self.action_space.low, self.action_space.high)


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    ou = OUStrategy(env.action_space, mu=0, theta=0.15, sigma=0.3)
    states = []
    for i in range(1000):
        states.append(ou.evolve_state()[0])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
