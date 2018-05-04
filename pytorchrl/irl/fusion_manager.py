import os
import joblib
import re

import numpy as np

class FusionDistrManager(object):
    def add_paths(self, paths):
        raise NotImplementedError()

    def sample_paths(self, n):
        raise NotImplementedError()


class RamFusionDistr(FusionDistrManager):
    def __init__(self, buf_size, subsample_ratio=0.5):
        self.buf_size = buf_size
        self.buffer = []
        self.subsample_ratio = subsample_ratio

    def add_paths(self, paths, subsample=True):
        if subsample:
            paths = paths[:int(len(paths)*self.subsample_ratio)]
        self.buffer.extend(paths)
        overflow = len(self.buffer)-self.buf_size
        while overflow > 0:
            #self.buffer = self.buffer[overflow:]
            N = len(self.buffer)
            probs = np.arange(N)+1
            probs = probs/float(np.sum(probs))
            pidx = np.random.choice(np.arange(N), p=probs)
            self.buffer.pop(pidx)
            overflow -= 1

    def sample_paths(self, n):
        if len(self.buffer) == 0:
            return []
        else:
            pidxs = np.random.randint(0, len(self.buffer), size=(n))
            return [self.buffer[pidx] for pidx in pidxs]


if __name__ == "__main__":
    #fm = DiskFusionDistr(path_dir='data_nobs/gridworld_random/gru1')
    #paths = fm.sample_paths(10)
    fm = RamFusionDistr(10)
    fm.add_paths([1,2,3,4,5,6,7,8,9,10,11,12,13])
    print(fm.buffer)
    print(fm.sample_paths(5))