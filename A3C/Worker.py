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
    worker_device = torch.device("cpu")
    print(f"Worker {worker_id} started on CPU.")

    # Worker-specific seeding
    torch.manual_seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

    local_env = gym.make(env_name, **ARGS)
    local_model = ActorCriticNetwork(n_observations, n_actions, observation_type, action_space_type).to(worker_device)

    local_model.load_state_dict(global_model.state_dict())
    local_model.train()

    state, _ = local_env.reset(seed=seed + worker_id)  # Initial state (returns obs, info)
    if observation_type == "discrete":
        state = torch.tensor(state, dtype=torch.long, device=worker_device)
    else:
        state = torch.tensor(state, dtype=torch.float32, device=worker_device)

    episode_reward = 0.0
    episode_length = 0
    episode_count = 0 

    try:
        while global_counter.value < max_global_steps and not stop_event.is_set():
            local_model.load_state_dict(global_model.state_dict())
            local_model.train() 

            # Storage for n-step rollout data
            log_probs_list: List[torch.Tensor] = []
            values_list: List[torch.Tensor] = [] 
            rewards_list: List[float] = []
            dones_list: List[float] = []
            entropies_list: List[torch.Tensor] = [] 

            episode_done_flag = False 

            # --- Rollout Phase (collect n steps or until episode ends) ---
            for step_idx in range(n_steps):

                state_tensor = state.to(worker_device)
                action_dist, value_pred = local_model(state_tensor)

                if action_space_type == "continuous":
                    action       = action_dist.sample()
                    log_prob     = action_dist.log_prob(action).sum(-1)
                    entropy      = action_dist.entropy().sum(-1)
                    action_np    = action.cpu().numpy().reshape(-1)
                    low, high    = local_env.action_space.low, local_env.action_space.high
                    action_env   = np.clip(action_np, low, high).astype(np.float32)

                else: 
                    action       = action_dist.sample()
                    log_prob     = action_dist.log_prob(action)
                    entropy      = action_dist.entropy()
                    action_env   = action.item()


                # Interact with environment
                next_state, reward, terminated, truncated, info = local_env.step(action_env)
                done = terminated or truncated 
                if observation_type == "discrete":
                    next_state = torch.tensor(next_state, dtype=torch.long, device=worker_device)
                else:
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=worker_device)

                log_probs_list.append(log_prob)
                values_list.append(value_pred)
                rewards_list.append(reward)
                dones_list.append(float(done))
                entropies_list.append(entropy)

                episode_reward += reward
                episode_length += 1
                state = next_state
                episode_done = done or (episode_length >= max_episode_steps)
                episode_done_flag = episode_done

                current_global_step = 0
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
                    if observation_type == "discrete":
                        state = torch.tensor(state, dtype=torch.long, device=worker_device)
                    else:
                        state = torch.tensor(state, dtype=torch.float32, device=worker_device)
                    episode_reward = 0.0
                    episode_length = 0
                    break

                if current_global_step >= max_global_steps:
                     stop_event.set()
                     break

            if not rewards_list:
                continue

            # Calculate bootstrap value V(s_{t+n})
            R = torch.tensor([0.0], dtype=torch.float32, device=worker_device)
            if not episode_done_flag and not stop_event.is_set(): 
                with torch.no_grad():
                    _, R = local_model(state.to(worker_device))

            # Compute n-step returns and advantages (function expects detached bootstrap value R)
            returns_tensor, advantages_tensor = compute_n_step_returns_advantages(
                rewards_list, values_list, R, dones_list, gamma
            )
            returns_tensor = returns_tensor.to(worker_device)
            advantages_tensor = advantages_tensor.to(worker_device)

            # --- Calculate Losses ---
            log_probs_tensor = torch.cat(log_probs_list)
            values_pred_tensor = torch.cat(values_list) 
            entropies_tensor = torch.cat(entropies_list)

            for tensor in [log_probs_tensor, values_pred_tensor, returns_tensor, advantages_tensor, entropies_tensor]:
                tensor = tensor.squeeze() 
                if tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)

            # for debugging
            # print(f"Worker {worker_id} Shapes: logp={log_probs_tensor.shape}, vals={values_pred_tensor.shape}, ret={returns_tensor.shape}, adv={advantages_tensor.shape}, ent={entropies_tensor.shape}")
            # assert log_probs_tensor.shape == values_pred_tensor.shape == returns_tensor.shape == advantages_tensor.shape == entropies_tensor.shape, "Shape mismatch before loss calculation!"

            # Actor loss
            policy_loss = -(log_probs_tensor * advantages_tensor.detach()).mean()
            # Critic loss -  MSE between *predicted* values and n-step returns
            value_loss = F.mse_loss(values_pred_tensor, returns_tensor.detach())
            entropy_loss = -entropies_tensor.mean()
            # TODO: maybe decay entropy only for some envs?
            progress_ratio = current_global_step / max_global_steps
            entropy_coeff_now =  max(entropy_coeff * (1.0 - progress_ratio), ENTROPY_MIN)
            # Combined loss
            total_loss = policy_loss + value_loss_coeff * value_loss + entropy_coeff_now * entropy_loss
            
            local_model.train()
            global_optimizer.zero_grad()
            local_model.zero_grad()
            total_loss.backward()
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=0.5)

            # Transfer gradients from local model to global model
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if global_param.grad is not None:
                    # Shouldn't happen if zero_grad was called correctly
                    global_param.grad = None # for safety
                if local_param.grad is not None:
                    global_param.grad = local_param.grad.clone().to(global_param.device)

            # Apply the gradients using the shared optimize
            global_optimizer.step()
            for param_group in global_optimizer.param_groups:
                # Adjust learning rate based on progress
                param_group["lr"] = LR_A3C * max((1.0 - progress_ratio), 0.001)
            if global_counter.value >= max_global_steps and not stop_event.is_set():
                print(f"Worker {worker_id} reached max global steps after update.")
                stop_event.set()


    except Exception as e:
        print(f"!!! Worker {worker_id} encountered an error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(("error", worker_id, str(e)))
        stop_event.set()

    finally:
        print(f"Worker {worker_id} finished. Total episodes: {episode_count}, Final Global steps: {global_counter.value}")
        result_queue.put(("finished", worker_id))
