import numpy as np
import torch

from pytorchrl.core.parameterized import Parameterized

class ImitationLearning(object):
    def __init__(self):
        pass

    @staticmethod
    def compute_path_probs(paths, policy_dist_type, insert=True, insert_key='a_logprobs'):
        """
        Returns a N * T matrix of action probabilities
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
        if stack:
            return [np.stack([t[key] for t in paths]).astype(np.float32) for key in keys]
        else:
            return [np.concatenate([t[key] for t in paths]).astype(np.float32) for key in keys]

    @staticmethod
    def sample_batch(*args, batch_size=32):
        N = args[0].shape[0]
        batch_idxs = np.random.randint(0, N, batch_size)  # trajectories are negatives
        return [data[batch_idxs] for data in args]

    def fit(self, paths, **kwargs):
        """
        Update the parameter of irl model
        """
        raise NotImplementedError()

    def eval(self, paths, **kwargs):
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def set_params(self, params):
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
        # TODO (ewei), add doc here
        """
        return ImitationLearning.extract_paths(paths, keys=keys, stack=stack)

    @staticmethod
    def unpack(data, paths):
        """
        # TODO (ewei), add doc here
        """
        lengths = [path['observations'].shape[0] for path in paths]
        unpacked = []
        idx = 0
        for l in lengths:
            unpacked.append(data[idx:idx+l])
            idx += l
        return unpacked


