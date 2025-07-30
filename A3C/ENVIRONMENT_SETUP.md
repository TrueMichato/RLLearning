# Environment Setup for A3C Implementation

This A3C implementation supports two environments from OpenAI Gymnasium:

## Supported Environments

### 1. CartPole-v1 (Default)

- **Description**: Balance a pole on a cart by moving the cart left or right
- **Action Space**: Discrete(2) - Push left (0) or Push right (1)
- **Observation Space**: Box(4) - [cart position, cart velocity, pole angle, pole angular velocity]
- **Goal**: Keep the pole upright for as long as possible
- **Success Criteria**: Episode reward of 500 (max episode length)
- **Installation**: No additional dependencies required

### 2. LunarLander-v3

- **Description**: Land a spacecraft on a landing pad using thrusters
- **Action Space**: Discrete(4) - Do nothing (0), Fire left engine (1), Fire main engine (2), Fire right engine (3)
- **Observation Space**: Box(8) - [x, y, vx, vy, angle, angular velocity, left leg contact, right leg contact]
- **Goal**: Land safely on the landing pad between the flags
- **Success Criteria**: Episode reward of 200+
- **Installation**: Requires Box2D physics engine

## Switching Between Environments

To switch environments, modify the `ENV_NAME` variable in `imports.py`:

```python
# For CartPole (default - no additional installation needed)
ENV_NAME = "CartPole-v1"

# For LunarLander (requires Box2D installation)
ENV_NAME = "LunarLander-v3"
```

## Installation Requirements

### For CartPole-v1 (Ready to use)

No additional packages needed. CartPole comes with the base gymnasium installation.

### For LunarLander-v3

You need to install Box2D physics engine:

#### Option 1: Install Box2D with conda (Recommended)

```bash
conda install -c conda-forge box2d-py
```

#### Option 2: Install with pip

```bash
# Install SWIG first (required for Box2D compilation)
# On Windows with conda:
conda install swig

# Then install gymnasium with box2d support
pip install "gymnasium[box2d]"
```

#### Option 3: Alternative pip installation

```bash
pip install box2d-py
```

## Usage

1. Make sure the desired environment is set in `imports.py`
2. Install required dependencies (if using LunarLander)
3. Run the training:

   ```bash
   python main.py
   ```

## Environment-Specific Settings

The code automatically adjusts certain parameters based on the selected environment:

- **Max Steps Per Episode**:
  - CartPole: 500 steps
  - LunarLander: 1000 steps
- **Network Architecture**: Automatically configured for the observation/action space dimensions
- **Plot Filenames**: Generated based on environment name (e.g., `a3c_cartpole_v1_training_progress.png`)

## Training Performance

### CartPole-v1

- **Training Time**: ~3-4 minutes for 100k steps
- **Expected Performance**: Should reach episode lengths of 200+ and eventually solve the environment (500 steps consistently)
- **Convergence**: Usually shows improvement within the first few thousand steps

### LunarLander-v3

- **Training Time**: ~10-15 minutes for 100k steps (more complex environment)
- **Expected Performance**: Should gradually improve from negative rewards to positive rewards (successful landings)
- **Convergence**: May take longer to show consistent improvement due to environment complexity

## Troubleshooting

### Box2D Installation Issues

If you encounter Box2D installation problems:

1. **Windows**: Try installing Microsoft Visual C++ Build Tools
2. **macOS**: Try `brew install swig` before pip installation
3. **Linux**: Install `swig` and `python3-dev` packages first

### Environment Not Found

If you get an environment registration error:

- Ensure gymnasium is up to date: `pip install --upgrade gymnasium`
- Check that the environment name is correct (case-sensitive)

### Memory/Performance Issues

If training is slow or crashes:

- Reduce `NUM_WORKERS` in `main.py` (try 4 or 8 workers instead of using all CPU cores)
- Reduce `MAX_GLOBAL_STEPS_A3C` for shorter training runs
- Consider using fewer rollout steps (`N_STEPS`)

## Example Training Output

**CartPole-v1:**

```
> Steps: 51250, Episodes: 2150, Avg Reward (last 50): 65.38, Avg Length (last 50): 65.4
```

**LunarLander-v3:**

```
> Steps: 25000, Episodes: 500, Avg Reward (last 50): -150.23, Avg Length (last 50): 180.5
```

## Customization

You can easily add support for other Gymnasium environments by:

1. Adding the environment name to the configuration in `imports.py`
2. Setting the correct observation and action space dimensions
3. Adjusting hyperparameters as needed for the specific environment

The code is designed to be easily extensible to other discrete action space environments.
