import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return
