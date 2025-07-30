from imports import *
from SharedActorCriticLogics import ActorCriticNetwork, compute_n_step_returns_advantages
from Worker import worker


if __name__ == "__main__":

    # Set multiprocessing start method (important for some OS like macOS/Windows)
    # 'spawn' is generally safer than 'fork' with CUDA/threading
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Could not set start method to 'spawn': {e}. Using default.")


    # --- Initialization ---
    # Initialize Global Network (on CPU, as share_memory works best)
    global_model_a3c = ActorCriticNetwork(n_observations, n_actions).to(device)
    # Crucial step: Ensure model parameters are shared across processes
    global_model_a3c.share_memory()
    print(f"Global model initialized on {device} and set to shared memory.")

    # Initialize Optimizer (acts on the shared global model's parameters)
    # Adam is common, but RMSprop was used in the original A3C paper
    global_optimizer_a3c = optim.RMSprop(global_model_a3c.parameters(), lr=LR_A3C, alpha=0.99, eps=1e-5)
    # global_optimizer_a3c = optim.Adam(global_model_a3c.parameters(), lr=LR_A3C)
    print(f"Global optimizer initialized: {type(global_optimizer_a3c).__name__}")

    # Shared counter for total steps taken across all workers
    global_step_counter = mp.Value('i', 0) # 'i' for integer, starts at 0

    # Manager for shared queue (needed for Queue to work across processes)
    manager = mp.Manager()
    result_queue = manager.Queue() # Queue for workers to send back results
    stop_event = manager.Event()   # Event to signal workers to stop

    # Lists for plotting overall progress (collected from queue)
    a3c_episode_rewards = []
    a3c_episode_lengths = []
    # all_worker_stats = {i: {"rewards": [], "lengths": []} for i in range(NUM_WORKERS)} # Optional detailed tracking

    print(f"\nStarting A3C Training with {NUM_WORKERS} workers...")
    print(f"Target Max Global Steps: {MAX_GLOBAL_STEPS_A3C}")
    print(f"Environment: {ENV_NAME}")
    print(f"Hyperparameters: Gamma={GAMMA_A3C}, LR={LR_A3C}, N_Steps={N_STEPS}, V_Coeff={VALUE_LOSS_COEFF_A3C}, E_Coeff={ENTROPY_COEFF_A3C}")

    start_time = time.time()

    # --- Create and Start Worker Processes ---
    workers = []
    for i in range(NUM_WORKERS):
        worker_process = mp.Process(target=worker,
                                    args=(i, global_model_a3c, global_optimizer_a3c,
                                          global_step_counter, MAX_GLOBAL_STEPS_A3C,
                                          ENV_NAME, N_STEPS, GAMMA_A3C,
                                          VALUE_LOSS_COEFF_A3C, ENTROPY_COEFF_A3C,
                                          MAX_STEPS_PER_EPISODE_A3C, result_queue,
                                          stop_event))
        workers.append(worker_process)
        worker_process.start()
        print(f"Worker {i} process started.")

    # --- Monitor Queue and Collect Results ---
    finished_workers = 0
    progress_updates = 0
    error_occurred = False

    while finished_workers < NUM_WORKERS:
        try:
            # Get result from the queue (blocks until item available)
            # Add a timeout to prevent hanging indefinitely if something goes wrong
            result = result_queue.get(timeout=120) # Increased timeout

            if isinstance(result, tuple):
                message_type = result[0]
                worker_id = result[1]

                if message_type == "episode_end":
                    ep_reward = result[2]
                    ep_length = result[3]
                    a3c_episode_rewards.append(ep_reward)
                    a3c_episode_lengths.append(ep_length)

                    # Print progress periodically based on total episodes collected
                    if len(a3c_episode_rewards) % 50 == 0:
                         avg_r = np.mean(a3c_episode_rewards[-50:]) if len(a3c_episode_rewards) >= 50 else np.mean(a3c_episode_rewards)
                         avg_l = np.mean(a3c_episode_lengths[-50:]) if len(a3c_episode_lengths) >= 50 else np.mean(a3c_episode_lengths)
                         print(f" > Steps: {global_step_counter.value}, Episodes: {len(a3c_episode_rewards)}, Avg Reward (last 50): {avg_r:.2f}, Avg Length (last 50): {avg_l:.1f}")

                elif message_type == "progress":
                    current_step = result[2]
                    progress_updates += 1
                    # Optional: Print less frequently
                    # if progress_updates % (NUM_WORKERS * 10) == 0:
                    #      print(f"   Progress Update: Global Steps ~{current_step}")

                elif message_type == "error":
                    error_msg = result[2]
                    print(f"!!! Received error from worker {worker_id}: {error_msg}")
                    error_occurred = True
                    if not stop_event.is_set():
                        print("   Signaling other workers to stop due to error.")
                        stop_event.set() # Signal all other workers to stop

                elif message_type == "finished":
                    print(f"Worker {worker_id} signaled completion.")
                    finished_workers += 1

            else:
                 print(f"Warning: Received unexpected item from queue: {result}")

        except queue.Empty:
            print("Warning: Result queue timed out. Checking worker status...")
            # Check if workers are still alive, maybe break if they are not
            alive_workers = sum(p.is_alive() for p in workers)
            print(f"Active workers: {alive_workers}/{NUM_WORKERS}")
            if alive_workers == 0 and finished_workers < NUM_WORKERS:
                 print("Error: All workers seem to have exited unexpectedly.")
                 if not stop_event.is_set(): stop_event.set() # Ensure stop is signaled
                 break # Exit monitoring loop
            if stop_event.is_set():
                 print("Stop event is set, likely due to error, max steps, or manual stop. Waiting for workers to finish.")
                 # Continue waiting for "finished" signals or timeout join later
                 # Break here might prevent collecting final "finished" signals
                 # Let the loop continue until finished_workers == NUM_WORKERS or join times out

        # Check if max steps reached globally, even if workers haven't signaled finish yet
        if not stop_event.is_set() and global_step_counter.value >= MAX_GLOBAL_STEPS_A3C:
            print(f"Max global steps ({MAX_GLOBAL_STEPS_A3C}) reached. Signaling workers to stop.")
            stop_event.set()

    # --- Wait for all Worker Processes to Finish ---
    print("\nWaiting for worker processes to join...")
    active_workers_final_check = []
    for i, p in enumerate(workers):
        p.join(timeout=30) # Add a reasonable timeout for join
        if p.is_alive():
            print(f"Warning: Worker {i} did not join cleanly after timeout. Terminating.")
            p.terminate() # Forcefully terminate if stuck
            active_workers_final_check.append(i)
        # else:
        #      print(f"Worker {i} joined successfully. Exit code: {p.exitcode}")


    end_time = time.time()
    print(f"\n--- {ENV_NAME} Training Finished (A3C) ---")
    print(f"Total global steps reached: {global_step_counter.value}")
    print(f"Total episodes completed: {len(a3c_episode_rewards)}")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    if error_occurred:
        print("Training finished DUE TO AN ERROR in one or more workers.")
    if active_workers_final_check:
        print(f"Workers {active_workers_final_check} had to be terminated.")

    # --- Plotting Results ---
    if a3c_episode_rewards:
        plt.figure(figsize=(12, 5))

        # Plot Episode Rewards
        plt.subplot(1, 2, 1)
        plt.plot(a3c_episode_rewards, label='Episode Reward', alpha=0.6)
        # Add a moving average
        if len(a3c_episode_rewards) >= 50:
            moving_avg_rewards = np.convolve(a3c_episode_rewards, np.ones(50)/50, mode='valid')
            plt.plot(np.arange(len(moving_avg_rewards)) + 49, moving_avg_rewards,
                     label='Moving Avg (50 ep)', color='red', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("A3C Episode Rewards over Time")
        plt.legend()
        plt.grid(True)

        # Plot Episode Lengths
        plt.subplot(1, 2, 2)
        plt.plot(a3c_episode_lengths, label='Episode Length', color='orange', alpha=0.6)
         # Add a moving average
        if len(a3c_episode_lengths) >= 50:
            moving_avg_lengths = np.convolve(a3c_episode_lengths, np.ones(50)/50, mode='valid')
            plt.plot(np.arange(len(moving_avg_lengths)) + 49, moving_avg_lengths,
                     label='Moving Avg (50 ep)', color='blue', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("A3C Episode Lengths over Time")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"a3c_{ENV_NAME.lower().replace('-', '_')}_training_progress.png")
        print(f"\nPlot saved to a3c_{ENV_NAME.lower().replace('-', '_')}_training_progress.png")
        # plt.show() # Uncomment to display plot directly
    else:
        print("\nNo episode data collected, skipping plotting.")

    print("Script finished.")