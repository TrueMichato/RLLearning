import torch.multiprocessing as mp # Use torch multiprocessing

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


# --- Hyperparameter Setup ---
GAMMA_A3C = 0.99             # Discount factor
LR_A3C = 1e-4                # Learning rate
N_STEPS = 5                  # Steps per update (n-step rollout length)
VALUE_LOSS_COEFF_A3C = 0.5   # Coefficient for value loss term
ENTROPY_COEFF_A3C = 0.01     # Coefficient for entropy bonus term

NUM_WORKERS = mp.cpu_count() # Use number of available CPU cores
# NUM_WORKERS = 4 # Or set manually
MAX_GLOBAL_STEPS_A3C = 1000  # Total training steps across all workers
MAX_STEPS_PER_EPISODE_A3C = 500 if ENV_NAME == "CartPole-v1" else 1000  # Environment-specific limits
