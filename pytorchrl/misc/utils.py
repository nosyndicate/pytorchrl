import numpy as np
import time


def gauss_log_pdf(params, x):
    mean, log_diag_std = params
    N, d = mean.shape
    cov =  np.square(np.exp(log_diag_std))
    diff = x-mean
    exp_term = -0.5 * np.sum(np.square(diff)/cov, axis=1)
    norm_term = -0.5*d*np.log(2*np.pi)
    var_term = -0.5 * np.sum(np.log(cov), axis=1)
    log_probs = norm_term + var_term + exp_term
    return log_probs #sp.stats.multivariate_normal.logpdf(x, mean=mean, cov=cov)

def categorical_log_pdf(params, x, one_hot=True):
    if not one_hot:
        raise NotImplementedError()
    probs = params[0]
    return np.log(np.max(probs * x, axis=1))


class TrainingIterator(object):
    def __init__(self, itrs, heartbeat=float('inf')):
        # float('inf') serves as unbounded upper value for comparison
        self.itrs = itrs
        self.heartbeat_time = heartbeat
        self.__vals = {}

    def random_idx(self, N, size):
        return np.random.randint(0, N, size=size)

    @property
    def itr(self):
        return self.__itr

    @property
    def heartbeat(self):
        return self.__heartbeat

    @property
    def elapsed(self):
        assert self.heartbeat, 'elapsed is only valid when heartbeat=True'
        return self.__elapsed

    def itr_message(self):
        return '==> Itr %d/%d (elapsed:%.2f)' % (self.itr+1, self.itrs, self.elapsed)

    def record(self, key, value):
        if key in self.__vals:
            self.__vals[key].append(value)
        else:
            self.__vals[key] = [value]

    def pop(self, key):
        vals = self.__vals.get(key, [])
        del self.__vals[key]
        return vals

    def pop_mean(self, key):
        return np.mean(self.pop(key))


    def __iter__(self):
        prev_time = time.time()
        self.__heartbeat = False
        for i in range(self.itrs):
            self.__itr = i
            cur_time = time.time()
            if (cur_time - prev_time) > self.heartbeat_time or (i == self.itrs - 1):
                self.__heartbeat = True
                self.__elapsed = cur_time - prev_time
                prev_time = cur_time
            yield self
            self.__heartbeat = False
