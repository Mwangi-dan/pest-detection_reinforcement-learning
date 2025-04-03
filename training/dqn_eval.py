import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.custom_env import FarmPestControl

def make_env():
    """Create the environment for evaluation."""
    return FarmPestControl(grid_size=(10, 10), num_pests=5, num_obstacles=10, max_steps_per_episode=200)

def evaluate_model(model_path, vec_normalize_path=None, num_episodes=10, render=True, delay=0.05):
    """
    Evaluate a trained model on the farm pest control environment.
    
    Args:
        model_path: Path to the saved model
        vec_normalize_path: Path to the saved VecNormalize statistics (if used during training)
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment during evaluation
        delay: Delay between frames when rendering (seconds)
    
    Returns:
        List of episode rewards and success rate
    """
    # Create and wrap the environment
    env = DummyVecEnv([make_env])
    
    # Load normalization parameters if provided
    if vec_normalize_path:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update normalization statistics
        env.norm_reward = False  # Don't normalize rewards during evaluation
    
    # Load the trained model
    model = DQN.load(model_path)
    
    episode_rewards = []
    success_count = 0
    steps_taken = []
    pests_eliminated = []
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        initial_pests = None
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        # Run episode until done
        while not done:
            # Get predicted action
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, rewards, dones, info = env.step(action)
            
            # Track first info to get initial pest count
            if step == 0:
                initial_pests = info[0].get('remaining_pests', 0)
            
            # Update tracking variables
            episode_reward += rewards[0]
            done = dones[0]
            step += 1
            
            # Print step info
            if info[0].get('pest_eliminated', False):
                print(f"Step {step}: Eliminated pest! Remaining: {info[0].get('remaining_pests', 0)}")
            if info[0].get('hit_obstacle', False):
                print(f"Step {step}: Hit obstacle! Total hits: {info[0].get('obstacles_hit', 0)}")
            
            # Render the environment
            if render:
                env.render()
                time.sleep(delay)
        
        # Episode stats
        is_success = info[0].get('remaining_pests', 0) == 0
        if is_success:
            success_count += 1
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        steps_taken.append(step)
        pests_eliminated.append(initial_pests - info[0].get('remaining_pests', 0))
        
        print(f"Episode {episode+1} complete:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {step}")
        print(f"  Pests eliminated: {pests_eliminated[-1]}/{initial_pests}")
        print(f"  Success: {'Yes' if is_success else 'No'}")
    
    # Calculate and print summary statistics
    success_rate = success_count / num_episodes
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(steps_taken)
    avg_pests = np.mean(pests_eliminated)
    
    print("\n--- Evaluation Summary ---")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Pests Eliminated: {avg_pests:.1f}")
    
    # Create visualization of results
    plot_evaluation_results(episode_rewards, steps_taken, pests_eliminated, success_rate)
    
    return episode_rewards, success_rate

def plot_evaluation_results(rewards, steps, pests, success_rate):
    """
    Create visualizations of the evaluation results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot rewards
    axes[0, 0].plot(rewards, marker='o')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Plot steps
    axes[0, 1].plot(steps, marker='o', color='green')
    axes[0, 1].set_title('Steps per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Plot pests eliminated
    axes[1, 0].plot(pests, marker='o', color='red')
    axes[1, 0].set_title('Pests Eliminated per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Pests Eliminated')
    axes[1, 0].grid(True)
    
    # Success rate pie chart
    labels = ['Success', 'Failure']
    sizes = [success_rate, 1 - success_rate]
    colors = ['#66b3ff', '#ff9999']
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].axis('equal')
    axes[1, 1].set_title('Success Rate')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.show()

if __name__ == "__main__":
    # Path to your trained model
    model_path = "dqn_farm_pest_models/final_model"
    
    # Path to normalization statistics (if you used VecNormalize)
    # vec_normalize_path = "dqn_farm_pest_logs/vec_normalize.pkl"
    
    # Run evaluation
    evaluate_model(
        model_path=model_path,
        num_episodes=10,
        render=True,
        delay=0.1
    )