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
from torch.distributions import Categorical
import torch.multiprocessing as mp # Use torch multiprocessing

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

# Environment configuration
# Choose either "CartPole-v1" or "LunarLander-v3"
# NOTE: For LunarLander-v3, you need to install Box2D: pip install "gymnasium[box2d]"
ENV_NAME = "CartPole-v1"  # Change to "LunarLander-v3" if desired (requires Box2D installation)

# Environment-specific parameters
if ENV_NAME == "CartPole-v1":
    n_observations = 4  # Cart position, cart velocity, pole angle, pole angular velocity
    n_actions = 2       # Push left, push right
elif ENV_NAME == "LunarLander-v3":
    n_observations = 8  # x, y, vx, vy, angle, angular velocity, left leg contact, right leg contact
    n_actions = 4       # Do nothing, fire left engine, fire main engine, fire right engine
else:
    raise ValueError(f"Unsupported environment: {ENV_NAME}")

print(f"Environment: {ENV_NAME}")
print(f"Observation space: {n_observations}")
print(f"Action space: {n_actions}")