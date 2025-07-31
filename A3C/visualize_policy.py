"""
Policy Visualization for A3C Training
Records videos/GIFs of the agent's performance at different training stages
"""
from imports import *
from SharedActorCriticLogics import ActorCriticNetwork
import cv2
from PIL import Image
import os

def evaluate_policy(model: ActorCriticNetwork, env_name: str, num_episodes: int = 5, 
                   render_mode: str = "rgb_array", seed: int = None) -> Tuple[float, List[np.ndarray]]:
    """
    Evaluate a policy and optionally record frames for video creation.
    
    Args:
        model: The actor-critic network to evaluate
        env_name: Name of the environment
        num_episodes: Number of episodes to run
        render_mode: Rendering mode ('rgb_array' for recording, 'human' for display)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (average_reward, list_of_frame_arrays)
    """
    # Create environment with rendering
    env = gym.make(env_name, **ARGS, render_mode=render_mode)

    model.eval()  # Set to evaluation mode
    
    episode_rewards = []
    all_frames = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode if seed else None)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        
        episode_reward = 0.0
        episode_frames = []
        done = False
        
        while not done:
            # Get frame if recording
            if render_mode == "rgb_array":
                frame = env.render()
                episode_frames.append(frame)
            
            # Get action from policy (no exploration - use mode)
            with torch.no_grad():
                action_dist, _ = model(state)
                action = action_dist.mode  # Use most likely action instead of sampling
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            episode_reward += reward
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
        
        episode_rewards.append(episode_reward)
        if render_mode == "rgb_array":
            all_frames.extend(episode_frames)
    
    env.close()
    model.train()  # Set back to training mode
    
    avg_reward = np.mean(episode_rewards)
    return avg_reward, all_frames

def save_frames_as_gif(frames: List[np.ndarray], filename: str, fps: int = 30, duration_per_frame: float = None):
    """
    Save a list of frames as an animated GIF.
    
    Args:
        frames: List of numpy arrays (RGB images)
        filename: Output filename
        fps: Frames per second (ignored if duration_per_frame is set)
        duration_per_frame: Duration per frame in milliseconds
    """
    if not frames:
        print("No frames to save!")
        return
    
    # Convert frames to PIL Images
    pil_frames = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        pil_frames.append(Image.fromarray(frame))
    
    # Calculate duration
    if duration_per_frame is None:
        duration_per_frame = 1000 // fps  # Convert fps to milliseconds per frame
    
    # Save as GIF
    pil_frames[0].save(
        filename,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_per_frame,
        loop=0
    )
    print(f"GIF saved: {filename}")

def save_frames_as_mp4(frames: List[np.ndarray], filename: str, fps: int = 30):
    """
    Save a list of frames as an MP4 video using OpenCV.
    
    Args:
        frames: List of numpy arrays (RGB images)
        filename: Output filename
        fps: Frames per second
    """
    if not frames:
        print("No frames to save!")
        return
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()
    print(f"MP4 saved: {filename}")

def record_policy_demonstration(model: ActorCriticNetwork, env_name: str, stage: str, 
                               output_dir: str = "policy_videos", format: str = "gif"):
    """
    Record a demonstration of the current policy.
    
    Args:
        model: The actor-critic network to demonstrate
        env_name: Name of the environment
        stage: Description of the training stage (e.g., "initial", "final")
        output_dir: Directory to save the recordings
        format: Output format ("gif", "mp4", or "both")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean environment name for filename
    clean_env_name = env_name.lower().replace("-", "_")
    
    print(f"\nRecording {stage} policy demonstration for {env_name}...")
    
    # Record the policy
    avg_reward, frames = evaluate_policy(
        model=model,
        env_name=env_name,
        num_episodes=3,  # Record 3 episodes
        render_mode="rgb_array",
        seed=42  # Use fixed seed for reproducibility
    )
    
    print(f"Average reward during recording: {avg_reward:.2f}")
    
    # Save in requested format(s)
    if format in ["gif", "both"]:
        gif_filename = os.path.join(output_dir, f"{clean_env_name}_{stage}_policy.gif")
        save_frames_as_gif(frames, gif_filename, fps=25)
    
    if format in ["mp4", "both"]:
        mp4_filename = os.path.join(output_dir, f"{clean_env_name}_{stage}_policy.mp4")
        save_frames_as_mp4(frames, mp4_filename, fps=25)
    
    return avg_reward

def create_policy_comparison(initial_model_path: str, final_model_path: str, env_name: str):
    """
    Create a side-by-side comparison of initial vs final policy.
    
    Args:
        initial_model_path: Path to the initial model state
        final_model_path: Path to the final model state  
        env_name: Name of the environment
    """
    print("\nCreating policy comparison...")
    
    # Load models
    initial_model = ActorCriticNetwork(n_observations, n_actions).to(device)
    final_model = ActorCriticNetwork(n_observations, n_actions).to(device)
    
    initial_model.load_state_dict(torch.load(initial_model_path, map_location=device))
    final_model.load_state_dict(torch.load(final_model_path, map_location=device))
    
    # Record both policies
    print("Recording initial policy...")
    initial_reward, initial_frames = evaluate_policy(initial_model, env_name, num_episodes=2, seed=42)
    
    print("Recording final policy...")  
    final_reward, final_frames = evaluate_policy(final_model, env_name, num_episodes=2, seed=42)
    
    print(f"Initial policy average reward: {initial_reward:.2f}")
    print(f"Final policy average reward: {final_reward:.2f}")
    print(f"Improvement: {final_reward - initial_reward:.2f}")
    
    # Save individual recordings
    clean_env_name = env_name.lower().replace("-", "_")
    save_frames_as_gif(initial_frames, f"policy_videos/{clean_env_name}_comparison_initial.gif")
    save_frames_as_gif(final_frames, f"policy_videos/{clean_env_name}_comparison_final.gif")
    
    return initial_reward, final_reward

if __name__ == "__main__":
    # Example usage
    print("Policy Visualization Module")
    print("This module provides functions to record and visualize A3C policies.")
    print("Use record_policy_demonstration() during training to capture policy evolution.")
