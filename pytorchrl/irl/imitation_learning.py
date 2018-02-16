import numpy as np
import torch

from pytorchrl.core.parameterized import Parameterized
from pytorchrl.misc.utils import gauss_log_pdf, categorical_log_pdf

DIST_GAUSSIAN = 'gaussian'
DIST_CATEGORICAL = 'categorical'

class ImitationLearning(object):
    def __init__(self):
        pass

    @staticmethod
    def compute_path_probs(paths, policy_dist_type=None, insert=True, insert_key='a_logprobs'):
        """
        Returns a N * T matrix of action probabilities, where N is number
        of trajectories, and T is the length of each trajectories.

        Parameters
        ----------
        paths (list): Each element is a dict. Each dict represent a whole
            trajectory, contains observations, actions, rewards, env_infos,
            agent_infos. observations and actions is of size T * dim, where T
            is the length of the trajectory, and dim is the dimension of observation
            or action. rewards is a vector of length T. agent_infos contains other
            information about the policy. For example, when we have a Gaussian policy,
            it may contain the means and log_stds of the Gaussian policy at each step.
        policy_dist_type (string): The distribution type
        insert (boolean): Whether to insert the action probabilities back into
            the paths
        insert_key (string): The key to be used when inserting back

        Returns
        -------
        action_probs (numpy.ndarray): The N * T numpy matrix, each element is
            the probability of the action at T-th timestep of N-th trajectory.
        """
        if policy_dist_type is None:
            # figure out the distribution type
            path0 = paths[0]
            if 'log_std' in path0['agent_infos']:
                pol_dist_type = DIST_GAUSSIAN
            elif 'prob' in path0['agent_infos']:
                pol_dist_type = DIST_CATEGORICAL
            else:
                raise NotImplementedError()

        # compute path probs
        num_path = len(paths)
        actions = [path['actions'] for path in paths]
        if pol_dist_type == DIST_GAUSSIAN:
            params = [(path['agent_infos']['mean'], path['agent_infos']['log_std']) for path in paths]
            path_probs = [gauss_log_pdf(params[i], actions[i]) for i in range(num_path)]
        elif pol_dist_type == DIST_CATEGORICAL:
            params = [(path['agent_infos']['prob'],) for path in paths]
            path_probs = [categorical_log_pdf(params[i], actions[i]) for i in range(num_path)]
        else:
            raise NotImplementedError("Unknown distribution type")

        if insert:
            for i, path in enumerate(paths):
                path[insert_key] = path_probs[i]

        return np.array(path_probs)

    @staticmethod
    def extract_paths(paths, keys=['observations', 'actions'], stack=True):
        """
        Put all the info in the paths into a single matrix. If stack is True,
        then we get a rank-3 tensor, N * T * dim, where the N is number of
        paths, T is the length of path, the dim is the dimension of either
        observation or action or something else.
        If stack is false, the trajectories will be concatenate together to
        form a very long trajectory, then we have a rank-2 matrix, where the
        first dimension is of N * T.

        Parameters
        ----------
        paths (list): See doc of compute_path_probs method.
        keys (list): list of string indicate the infos we want to extract
        stack (Boolean): Whether stack the data for concatenate them

        Returns
        -------
        matrix (numpy.ndarray): matrix described above.
        """
        if stack:
            return [np.stack([t[key] for t in paths]).astype(np.float32) for key in keys]
        else:
            return [np.concatenate([t[key] for t in paths]).astype(np.float32) for key in keys]

    @staticmethod
    def sample_batch(*args, batch_size=32):
        """
        Sample a batch of size batch_size from data.
        """
        N = args[0].shape[0]
        batch_idxs = np.random.randint(0, N, batch_size)  # trajectories are negatives
        return [data[batch_idxs] for data in args]

    def fit(self, paths, **kwargs):
        """
        Train the discriminator
        """
        raise NotImplementedError()

    def eval(self, paths, **kwargs):
        raise NotImplementedError()

    def get_params(self):
        """
        Returns the parameters of the discriminator
        """
        raise NotImplementedError()

    def set_params(self, params):
        """
        Set the parameters of the discriminator
        """
        raise NotImplementedError()

class TrajectoryIRL(ImitationLearning):
    """
    Base class for models that score entire trajectories at once.
    This method take a whole trajectories as one sample.
    """

    @property
    def score_trajectories(self):
        return True

class SingleTimestepIRL(ImitationLearning):
    """
    Base class for models that score single timesteps at once.
    This method take a single state action pair as one sample.
    SingleTimestepIRL is better than TrajectoryIRL at that it has lower variance
    in estimation.
    """
    @property
    def score_trajectories(self):
        return False

    @staticmethod
    def extract_paths(paths, keys=('observations', 'actions'), stack=False):
        """
        See doc of extract_paths in ImitationLearning class
        """
        return ImitationLearning.extract_paths(paths, keys=keys, stack=stack)

    @staticmethod
    def unpack(data, paths):
        """
        Chop the data into smaller piece according to the info in paths.
        Each of the smaller piece of data should have the same length
        as the observations in the corresponding paths. Thus, data should
        have the length which is the summation of the all the length of
        observations in paths.

        Parameters
        ----------
        data (numpy.ndarray): A vector contains the data of size of the
            summation of length of all the observations in paths
        paths (list): See doc of compute_path_probs method

        Returns
        -------
        unpacked (list): Each element is a numpy array which is a smaller
            piece of the data. The rule to divide them is described above.
        """
        lengths = [path['observations'].shape[0] for path in paths]

        unpacked = []
        idx = 0
        for l in lengths:
            unpacked.append(data[idx:idx+l])
            idx += l

        return unpacked


