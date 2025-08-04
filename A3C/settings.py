import torch.multiprocessing as mp # Use torch multiprocessing

# Environment configuration
# Choose either "CartPole-v1" or "LunarLander-v3"
# NOTE: For LunarLander-v3, you need to install Box2D: pip install "gymnasium[box2d]"
ENV_NAME = "CartPole-v1"  # Change to "LunarLander-v3" if desired (requires Box2D installation)

# Environment-specific parameters
if ENV_NAME == "CartPole-v1":
    n_observations = 4  # Cart position, cart velocity, pole angle, pole angular velocity
    n_actions = 2       # Push left, push right
    observation_type = "continuous"  # Vector observations
    action_space_type = "discrete"  # Discrete action space (0 or 1)
    # --- Hyperparameter Setup ---
    GAMMA_A3C = 0.99             # Discount factor
    LR_A3C = 1e-4                # Learning rate
    N_STEPS = 5                  # Steps per update (n-step rollout length)
    VALUE_LOSS_COEFF_A3C = 0.5   # Coefficient for value loss term
    ENTROPY_COEFF_A3C = 0.01     # Coefficient for entropy bonus term
    MAX_GLOBAL_STEPS_A3C = 100000  # Total training steps across all workers
    MAX_STEPS_PER_EPISODE_A3C = 500

    # --- Environment Arguments --
    ARGS = dict()

elif ENV_NAME == "LunarLander-v3":
    n_observations = 8  # x, y, vx, vy, angle, angular velocity, left leg contact, right leg contact
    n_actions = 4       # Do nothing, fire left engine, fire main engine, fire right engine
    observation_type = "continuous"  # Vector observations
    action_space_type = "discrete"  # Discrete action space (0-3)
    #     action_space_type = "continuous"
    # --- Hyperparameter Setup ---
    GAMMA_A3C = 0.99             # Discount factor
    LR_A3C = 8e-4                # Higher learning rate for faster learning
    N_STEPS = 20                 # Longer rollouts for better credit assignment
    VALUE_LOSS_COEFF_A3C = 1.0   # Higher value loss coefficient for better bootstrapping
    ENTROPY_COEFF_A3C = 0.01    # Lower entropy coefficient for less random exploration
    MAX_GLOBAL_STEPS_A3C = 5000000  # More training steps needed for complex environment
    MAX_STEPS_PER_EPISODE_A3C = 10000

    # --- Environment Arguments --
    ARGS = {"continuous":False, 
            # "gravity":-10.0,
            # "enable_wind":False, 
            # "wind_power":15.0, 
            # "turbulence_power":1.5
               }

elif ENV_NAME == "FrozenLake-v1":
    n_observations = 16  # 4x4 grid, 16 possible states (0-15)
    n_actions = 4        # Up, down, left, right
    observation_type = "discrete"  # Single integer observations (state index)
    action_space_type = "discrete"  # Discrete action space (0-3)
    # --- Hyperparameter Setup ---
    GAMMA_A3C = 0.99             # Discount factor
    LR_A3C = 1e-3                # Higher learning rate for discrete environments
    N_STEPS = 5                  # Steps per update (n-step rollout length)
    VALUE_LOSS_COEFF_A3C = 0.5   # Coefficient for value loss term
    ENTROPY_COEFF_A3C = 0.1      # Higher entropy for exploration in sparse reward env
    MAX_GLOBAL_STEPS_A3C = 200000  # More steps needed for sparse reward environments
    MAX_STEPS_PER_EPISODE_A3C = 100

    # --- Environment Arguments ---
    ARGS = {"desc":None, 
            "map_name":"4x4", 
            "is_slippery":False}  # Non-slippery for easier learning

elif ENV_NAME == "InvertedPendulum-v5":
    n_observations = 4  # x, x_dot, theta
    n_actions = 1  # Force applied to the pendulum
    observation_type = "continuous"  # Vector observations
    action_space_type = "continuous"  # Continuous action space (force magnitude)
    # --- Hyperparameter Setup ---
    GAMMA_A3C = 0.99             # Discount factor
    LR_A3C = 1e-4                # Learning rate
    N_STEPS = 5                  # Steps per update (n-step rollout length)
    VALUE_LOSS_COEFF_A3C = 0.5   # Coefficient for value loss term
    ENTROPY_COEFF_A3C = 0.01     # Coefficient for entropy bonus term
    MAX_GLOBAL_STEPS_A3C = 200000  # Total training steps across all workers
    MAX_STEPS_PER_EPISODE_A3C = 1000
    # --- Environment Arguments ---
    ARGS = {"reset_noise_scale":0.1}  # Add noise to initial state for exploration

else:
    raise ValueError(f"Unsupported environment: {ENV_NAME}")


ENTROPY_MIN = 0.0001

print(f"Environment: {ENV_NAME}")
# print(f"Observation space: {gym}")
# print(f"Action space: {n_actions}")



NUM_WORKERS = mp.cpu_count() # Use number of available CPU cores
# NUM_WORKERS = 4 # Or set manually


# Policy Recording Settings
RECORD_INITIAL_POLICY = True   # Record policy at the start of training
RECORD_FINAL_POLICY = True     # Record policy at the end of training
RECORD_INTERMEDIATE = True     # Record policy at 25%, 50%, 75% progress
VIDEO_FORMAT = "gif"           # "gif", "mp4", or "both"