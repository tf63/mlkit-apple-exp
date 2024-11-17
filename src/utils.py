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


def save_images(images, image_name):
    for i, image in enumerate(images):
        out_dir = os.path.join('out', image_name)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f'{image_name}_{i}.png')
        image.save(out_path)

        print(f'save to {out_path}')


def prepare(seed=0):
    make_out_dir()
    fix_seed(seed)
