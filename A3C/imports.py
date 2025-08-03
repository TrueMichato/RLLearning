# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple
from itertools import count
from typing import List, Tuple, Dict, Optional, Callable
import time
import queue
import gymnasium as gym

# Import PyTorch and multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp # Use torch multiprocessing

from settings import *

# Set up device (Workers likely run on CPU, global model might be GPU but requires care)
# For simplicity, let's assume CPU for this example to avoid GPU sharing complexities.
device = torch.device("cpu") 
print(f"Using device: {device}")

# Set random seeds for reproducibility in the main process
# Note: workers will need their own seeding if full reproducibility is needed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)