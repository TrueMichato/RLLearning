from imports import *
from SharedActorCriticLogics import ActorCriticNetwork, compute_n_step_returns_advantages

def worker(worker_id: int,
           global_model: ActorCriticNetwork,
           global_optimizer: optim.Optimizer,
           global_counter: mp.Value, # Shared counter for total steps
           max_global_steps: int,
           env_name: str,  # Environment name (e.g., "CartPole-v1" or "LunarLander-v3")
           n_steps: int, # N-step rollout length
           gamma: float,
           value_loss_coeff: float,
           entropy_coeff: float,
           max_episode_steps: int, # Max steps per episode in env
           result_queue: mp.Queue, # Queue for sending results (reward, length, errors, progress)
           stop_event: mp.Event) -> None: # Event to signal workers to stop
    """
    Function executed by each A3C worker process. (Corrected gradient handling)
    """
    worker_device = torch.device("cpu") # Workers run on CPU
    print(f"Worker {worker_id} started on CPU.")

    # Worker-specific seeding
    torch.manual_seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

    # Create local environment and model
    local_env = gym.make(env_name, **ARGS)
    local_model = ActorCriticNetwork(n_observations, n_actions, observation_type, action_space_type).to(worker_device)

    # Initial sync with global model
    local_model.load_state_dict(global_model.state_dict())
    # Keep model in train() mode as default unless BatchNorm/Dropout are used
    local_model.train()

    state, _ = local_env.reset(seed=seed + worker_id)  # Initial state (returns obs, info)
    # Convert to tensor with appropriate dtype based on observation type
    if observation_type == "discrete":
        state = torch.tensor(state, dtype=torch.long, device=worker_device)
    else:
        state = torch.tensor(state, dtype=torch.float32, device=worker_device)

    episode_reward = 0.0
    episode_length = 0
    episode_count = 0 # Track episodes completed by this worker

    try:
        while global_counter.value < max_global_steps and not stop_event.is_set():
            # --- Sync local model with global model before each rollout ---
            local_model.load_state_dict(global_model.state_dict())
            local_model.train() # Ensure it's in train mode

            # Storage for n-step rollout data
            log_probs_list: List[torch.Tensor] = []
            values_list: List[torch.Tensor] = [] # Store tensors with grad_fn
            rewards_list: List[float] = []
            dones_list: List[float] = []
            entropies_list: List[torch.Tensor] = [] # Store tensors with grad_fn

            episode_done_flag = False # Track if episode finished within this rollout

            # --- Rollout Phase (collect n steps or until episode ends) ---
            for step_idx in range(n_steps):
                # Ensure state tensor is on the correct device (CPU)
                state_tensor = state.to(worker_device)

                # >>>>> Perform forward pass *without* torch.no_grad() <<<<<
                action_dist, value_pred = local_model(state_tensor)

                if action_space_type == "continuous":
                    # Sample continuous action, sum log-prob over dims
                    action       = action_dist.sample()
                    log_prob     = action_dist.log_prob(action).sum(-1)
                    entropy      = action_dist.entropy().sum(-1)
                    # Convert to numpy and clip to env bounds
                    action_np    = action.cpu().numpy().reshape(-1)
                    low, high    = local_env.action_space.low, local_env.action_space.high
                    action_env   = np.clip(action_np, low, high).astype(np.float32)

                else:  # discrete
                    action       = action_dist.sample()
                    log_prob     = action_dist.log_prob(action)
                    entropy      = action_dist.entropy()
                    action_env   = action.item()


                # Interact with environment
                next_state, reward, terminated, truncated, info = local_env.step(action_env)
                done = terminated or truncated  # Episode ends if terminated or truncated
                
                
                # Convert next_state to tensor with appropriate dtype
                if observation_type == "discrete":
                    next_state = torch.tensor(next_state, dtype=torch.long, device=worker_device)
                else:
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=worker_device)

                # Store transition data (log_prob, value_pred, entropy now have grad history)
                log_probs_list.append(log_prob)
                values_list.append(value_pred) # value_pred has grad_fn
                rewards_list.append(reward)
                dones_list.append(float(done))
                entropies_list.append(entropy) # entropy has grad_fn

                episode_reward += reward
                episode_length += 1

                # Update state
                state = next_state

                # Check for termination conditions
                episode_done = done or (episode_length >= max_episode_steps)
                episode_done_flag = episode_done # Store if episode finished in this loop

                # Increment global step counter safely
                current_global_step = 0 # Initialize
                with global_counter.get_lock():
                    global_counter.value += 1
                    current_global_step = global_counter.value

                # Optional: Send periodic progress update
                if current_global_step > 0 and current_global_step % 5000 == 0:
                    result_queue.put(("progress", worker_id, current_global_step))

                # If episode ended, reset env and log results
                if episode_done:
                    episode_count += 1
                    result_queue.put(("episode_end", worker_id, episode_reward, episode_length))
                    state, _ = local_env.reset(seed=seed + worker_id + episode_count)  # Reset with new seed
                    # Convert to tensor with appropriate dtype based on observation type
                    if observation_type == "discrete":
                        state = torch.tensor(state, dtype=torch.long, device=worker_device)
                    else:
                        state = torch.tensor(state, dtype=torch.float32, device=worker_device)
                    episode_reward = 0.0
                    episode_length = 0
                    break # End inner rollout loop

                # Check global step limit inside inner loop too
                if current_global_step >= max_global_steps:
                     stop_event.set() # Signal stop if limit reached
                     break

            # --- Prepare for Gradient Calculation ---
            if not rewards_list: # Skip if rollout was empty
                continue

            # Calculate bootstrap value V(s_{t+n})
            R = torch.tensor([0.0], dtype=torch.float32, device=worker_device)
            if not episode_done_flag and not stop_event.is_set(): # If the episode did *not* end and not stopping
                # Use the local model to estimate value of the *next* state
                # Use torch.no_grad() here, as this is a target value
                with torch.no_grad():
                    _, R = local_model(state.to(worker_device)) # state is the state after the loop

            # Compute n-step returns and advantages (function expects detached bootstrap value R)
            # values_list contains tensors with grad_fn, compute_n_step will detach them internally for advantage calc
            returns_tensor, advantages_tensor = compute_n_step_returns_advantages(
                rewards_list, values_list, R, dones_list, gamma
            )

            # Move targets to the correct device (they are created on CPU)
            returns_tensor = returns_tensor.to(worker_device)
            advantages_tensor = advantages_tensor.to(worker_device)

            # --- Calculate Losses ---
            # Convert lists of tensors (which have grad_fn) to single tensors
            log_probs_tensor = torch.cat(log_probs_list)
            values_pred_tensor = torch.cat(values_list) # These are the V(s_t) predictions with grad_fn
            entropies_tensor = torch.cat(entropies_list)

            # Squeeze tensors ONLY if they have extra dims (e.g., shape [N, 1] -> [N])
            # Ensure the dimensions match for calculation.
            # Common shapes: log_probs [N], values [N], returns [N], advantages [N], entropies [N]
            log_probs_tensor = log_probs_tensor.squeeze()
            values_pred_tensor = values_pred_tensor.squeeze()
            returns_tensor = returns_tensor.squeeze()
            advantages_tensor = advantages_tensor.squeeze()
            entropies_tensor = entropies_tensor.squeeze()

            # Ensure tensors are at least 1D after squeeze (handles n_steps=1 case)
            if log_probs_tensor.dim() == 0: log_probs_tensor = log_probs_tensor.unsqueeze(0)
            if values_pred_tensor.dim() == 0: values_pred_tensor = values_pred_tensor.unsqueeze(0)
            if returns_tensor.dim() == 0: returns_tensor = returns_tensor.unsqueeze(0)
            if advantages_tensor.dim() == 0: advantages_tensor = advantages_tensor.unsqueeze(0)
            if entropies_tensor.dim() == 0: entropies_tensor = entropies_tensor.unsqueeze(0)

            # Final shape check before loss (helpful for debugging)
            # print(f"Worker {worker_id} Shapes: logp={log_probs_tensor.shape}, vals={values_pred_tensor.shape}, ret={returns_tensor.shape}, adv={advantages_tensor.shape}, ent={entropies_tensor.shape}")
            # assert log_probs_tensor.shape == values_pred_tensor.shape == returns_tensor.shape == advantages_tensor.shape == entropies_tensor.shape, "Shape mismatch before loss calculation!"


            # Actor loss (Policy Gradient) - detach advantages
            policy_loss = -(log_probs_tensor * advantages_tensor.detach()).mean()

            # Critic loss (Value Function) - MSE between *predicted* values and n-step returns
            value_loss = F.mse_loss(values_pred_tensor, returns_tensor.detach())

            # Entropy bonus - mean entropy over the batch
            entropy_loss = -entropies_tensor.mean()

            progress_ratio = current_global_step / max_global_steps
            # Linearly decay entropy coefficient (e.g. from ENTROPY_COEFF to 0.001)
            entropy_coeff_now = np.max(entropy_coeff * (1.0 - progress_ratio), ENTROPY_MIN)

            # Combined loss
            total_loss = policy_loss + value_loss_coeff * value_loss + entropy_coeff_now * entropy_loss

            # --- Compute Gradients and Update Global Network ---
            # Ensure model is in training mode for backward pass
            local_model.train()

            # Zero gradients of the *global* optimizer/model before local calculation
            global_optimizer.zero_grad()
            local_model.zero_grad()

            # Calculate gradients for the local model based on the total loss
            total_loss.backward()

            # Optional: Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=0.5)

            # Transfer gradients from local model to global model
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if global_param.grad is not None:
                    # This shouldn't happen if zero_grad was called correctly, but indicates potential issues
                    # print(f"Warning: Worker {worker_id} - Global grad already exists for {name}")
                    # For safety, let's zero it before copying, although ideally optimizer.zero_grad() handles this.
                    global_param.grad = None

                if local_param.grad is not None:
                    # Clone grad from local to global param's .grad attribute
                    global_param.grad = local_param.grad.clone().to(global_param.device) # Ensure grad is on global model's device

            # Apply the gradients using the shared optimizer (updates global model)
            global_optimizer.step()
            for param_group in global_optimizer.param_groups:
                param_group["lr"] = LR_A3C *  np.max((1.0 - progress_ratio), 0.001)

            # Check if max global steps reached after update
            if global_counter.value >= max_global_steps and not stop_event.is_set():
                print(f"Worker {worker_id} reached max global steps after update.")
                stop_event.set() # Signal others to stop


    except Exception as e:
        print(f"!!! Worker {worker_id} encountered an error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(("error", worker_id, str(e)))
        stop_event.set() # Signal others to stop

    finally:
        print(f"Worker {worker_id} finished. Total episodes: {episode_count}, Final Global steps: {global_counter.value}")
        result_queue.put(("finished", worker_id))
