from imports import *

class ActorCriticNetwork(nn.Module):
    """ Combined Actor-Critic network for A3C """
    def __init__(self, n_observations: int, n_actions: int):
        super(ActorCriticNetwork, self).__init__()
        # Shared layers
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)

        # Actor head (outputs action logits)
        self.actor_head = nn.Linear(128, n_actions)

        # Critic head (outputs state value)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Forward pass, returns action distribution and state value.

        Parameters:
        - x (torch.Tensor): Input state tensor. Should be on the correct device.

        Returns:
        - Tuple[Categorical, torch.Tensor]:
            - Action distribution (Categorical).
            - State value estimate (Tensor).
        """
        # Ensure input is a FloatTensor
        if not isinstance(x, torch.Tensor):
             # Assume input needs conversion, place on model's device implicitly? No, safer to manage explicitly.
             # Let's assume x is already a tensor on the correct device (CPU for workers)
             raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        elif x.dtype != torch.float32:
             x = x.to(dtype=torch.float32)

        # Add batch dimension if missing (e.g., single state input)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Shared layers
        x = F.relu(self.layer1(x))
        shared_features = F.relu(self.layer2(x))

        # Actor head
        action_logits = self.actor_head(shared_features)
        # Ensure logits are on the same device before creating Categorical
        action_dist = Categorical(logits=action_logits.to(x.device))

        # Critic head
        state_value = self.critic_head(shared_features)

        # If input had no batch dim, remove it from output value
        if x.shape[0] == 1 and state_value.dim() > 0: # Check state_value dim > 0 before squeeze
            state_value = state_value.squeeze(0)

        return action_dist, state_value
    
def compute_n_step_returns_advantages(rewards: List[float],
                                      values: List[torch.Tensor],
                                      bootstrap_value: torch.Tensor,
                                      dones: List[float],
                                      gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes n-step returns (targets for critic) and advantages for actor.

    Parameters:
    - rewards (List[float]): List of rewards from the n-step rollout.
    - values (List[torch.Tensor]): List of value estimates V(s_t) for the rollout steps (as tensors, with grad history).
    - bootstrap_value (torch.Tensor): Value estimate V(s_{t+n}) for bootstrapping. Should be detached.
    - dones (List[float]): List of done flags (0.0 for not done, 1.0 for done).
    - gamma (float): Discount factor.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]:
        - n_step_returns: Target values for the critic (on CPU).
        - advantages: Advantage estimates for the actor (on CPU).
    """
    n_steps = len(rewards)
    # Detach values *here* for calculation, they retain grad_fn upstream for loss
    values_tensor = torch.cat([v.detach() for v in values]).squeeze().to(torch.device("cpu"))

    # Detach bootstrap value as well and ensure it's on CPU
    R = bootstrap_value.detach().to(torch.device("cpu"))

    # Initialize tensors on CPU (as workers run on CPU)
    returns = torch.zeros(n_steps, dtype=torch.float32, device=torch.device("cpu"))
    advantages = torch.zeros(n_steps, dtype=torch.float32, device=torch.device("cpu"))

    # Calculate backwards from the last step
    for t in reversed(range(n_steps)):
        # R is the discounted return from step t onwards
        # If done[t] is 1.0, the state t+1 was terminal, so its value is 0.
        R = rewards[t] + gamma * R * (1.0 - dones[t])
        returns[t] = R

        # Advantage A_t = n_step_return(R_t) - V(s_t) (using detached value here)
        # Ensure values_tensor has the right shape if n_steps=1
        value_t = values_tensor if values_tensor.dim() == 0 else values_tensor[t]
        advantages[t] = R - value_t

    # Standardization of advantages is often helpful but omitted here for simplicity
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages