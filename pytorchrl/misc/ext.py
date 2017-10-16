import random
import numpy as np

from rllab.misc.console import colorize

def set_seed(seed):
    seed %= 4294967294
    global seed_
    seed_ = seed
    import lasagne
    random.seed(seed)
    np.random.seed(seed)
    lasagne.random.set_rng(np.random.RandomState(seed))

    # Set random seed if tensorflow is in use
    try:
        import tensorflow as tf
        tf.set_random_seed(seed)
    except Exception as e:
        print(e)

    # Set random seed if pytorch is in use
    try:
        import torch
        torch.manual_seed(seed)
    except Exception as e:
        print(e)

    print((
        colorize(
            'using seed %s' % (str(seed)),
            'green'
        )
    ))