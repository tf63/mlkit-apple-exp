import os
import random

import numpy as np
import torch


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def make_out_dir():
    os.makedirs('out', exist_ok=True)


def prepare(seed=0):
    make_out_dir()
    fix_seed(seed)
