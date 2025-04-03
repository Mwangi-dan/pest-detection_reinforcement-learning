import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.custom_env import FarmPestControl

def make_env():
    """Create environment for evaluation."""
    return FarmPestControl(grid_size=(10, 10), num_pests=5, num_obstacles=10, max_steps_per_episode=200)

def evaluate_ppo_model(model_path, vec_normalize_path=None, num_episodes=10, render=True):
    """
    Evaluate a trained PPO model on the farm pest control environment.
    
    Args:
        model_path: Path to the saved model
        vec_normalize_path: Path to saved normalization parameters
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    """
    # Create evaluation environment
    env = DummyVecEnv([make_env])
    
    # Load normalization parameters if provided
    if vec_normalize_path:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Track metrics
    episode_rewards = []
    episode_steps = []
    episode_pests_eliminated = []
    success_count = 0
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        pests_at_start = None
        
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, rewards, dones, infos = env.step(action)
            
            # Track initial pest count
            if steps == 0:
                pests_at_start = infos[0].get('remaining_pests', 0)
                
            # Update metrics
            episode_reward += rewards[0]
            done = dones[0]
            steps += 1
            
            # Display information about important events
            if infos[0].get('pest_eliminated', False):
                print(f"Episode {episode+1}, Step {steps}: Eliminated pest! Remaining: {infos[0].get('remaining_pests', 0)}")
                
            if infos[0].get('hit_obstacle', False):
                print(f"Episode {episode+1}, Step {steps}: Hit obstacle")
                
            # Render if specified
            if render:
                env.render()
                time.sleep(0.05)  # Short delay to make visualization visible
                
        # Episode complete
        pests_eliminated = pests_at_start - infos[0].get('remaining_pests', 0)
        success = infos[0].get('remaining_pests', 0) == 0
        
        if success:
            success_count += 1
            
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_pests_eliminated.append(pests_eliminated)
        
        # Print episode summary
        print(f"\nEpisode {episode+1}/{num_episodes} complete:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps Taken: {steps}")
        print(f"  Pests Eliminated: {pests_eliminated}/{pests_at_start}")
        print(f"  Success: {'Yes' if success else 'No'}")
        
    # Calculate summary statistics
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    avg_pests = np.mean(episode_pests_eliminated)
    success_rate = success_count / num_episodes
    
    # Print evaluation summary
    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Pests Eliminated: {avg_pests:.1f}")
    
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot rewards
    axs[0, 0].plot(episode_rewards, 'b-o')
    axs[0, 0].set_title('Rewards per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    axs[0, 0].grid(True)
    
    # Plot steps
    axs[0, 1].plot(episode_steps, 'g-o')
    axs[0, 1].set_title('Steps per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Steps')
    axs[0, 1].grid(True)
    
    # Plot pests eliminated
    axs[1, 0].plot(episode_pests_eliminated, 'r-o')
    axs[1, 0].set_title('Pests Eliminated per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Number of Pests')
    axs[1, 0].grid(True)
    
    # Plot success pie chart
    labels = ['Success', 'Failure']
    sizes = [success_rate, 1 - success_rate]
    colors = ['green', 'red']
    axs[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axs[1, 1].axis('equal')
    axs[1, 1].set_title('Success Rate')
    
    plt.tight_layout()
    plt.savefig('ppo_evaluation_results.png')
    plt.show()
    
    return {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'pests_eliminated': episode_pests_eliminated,
        'success_rate': success_rate
    }

if __name__ == "__main__":
    # Path to your trained model and normalization stats
    model_path = "ppo_farm_pest_models/final_model"
    vec_normalize_path = "ppo_farm_pest_logs/vec_normalize.pkl"
    
    # Run evaluation
    results = evaluate_ppo_model(
        model_path=model_path,
        vec_normalize_path=vec_normalize_path,
        num_episodes=10,
        render=True
    )