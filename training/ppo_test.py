import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from environment.custom_env import FarmPestControl  # Import the custom environment

# Set random seed for reproducibility
RANDOM_SEED = 42
set_random_seed(RANDOM_SEED)

# Create log directory
log_dir = "ppo_farm_pest_logs/"
model_dir = "ppo_farm_pest_models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def make_env(rank, seed=0):
    """Create environment for training."""
    def _init():
        env = FarmPestControl(grid_size=(10, 10), num_pests=5, num_obstacles=10, max_steps_per_episode=200)
        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _init

# Create the vectorized environment
env = DummyVecEnv([make_env(i) for i in range(1)])

# Normalize observations and rewards
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

# Define hyperparameters appropriate for PPO with this environment
ppo_params = {
    "learning_rate": 3e-4,       # Learning rate - standard for PPO
    "n_steps": 2048,             # Steps to collect before updating policy - typical for PPO
    "batch_size": 64,            # Minibatch size - balanced for training stability and speed
    "n_epochs": 10,              # Number of epochs to update policy for each batch - standard for PPO
    "gamma": 0.99,               # Discount factor - standard for most RL tasks
    "gae_lambda": 0.95,          # GAE lambda parameter - helps with variance reduction
    "clip_range": 0.2,           # Clipping parameter for PPO - standard value
    "clip_range_vf": None,       # No clipping of value function
    "ent_coef": 0.01,            # Entropy coefficient - small value to encourage exploration
    "vf_coef": 0.5,              # Value function coefficient - balanced for policy and value learning
    "max_grad_norm": 0.5,        # Max gradient norm - helps with training stability
    "use_sde": False,            # Don't use state-dependent exploration
    "sde_sample_freq": -1,       # Not used since use_sde is False
}

# Create the policy
policy_kwargs = dict(
    net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Separate networks for policy and value function
)

# Create the PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=policy_kwargs,
    **ppo_params
)

# Setup callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=model_dir,
    name_prefix="ppo_farm_pest"
)

# Total timesteps for training
total_timesteps = 70000  # Adjust based on convergence

# Train the agent
print("Starting PPO training...")
model.learn(
    total_timesteps=total_timesteps,
    callback=checkpoint_callback,
    progress_bar=True,
)

# Save the final model
model.save(f"{model_dir}/final_model")
env.save(f"{log_dir}/vec_normalize.pkl")
print(f"Training complete. Final model saved to {model_dir}/final_model")

# Function to plot training progress
def plot_results():
    """Plot the training progress."""
    plt.figure(figsize=(10, 6))
    
    # Load results
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    
    # Plot learning curve
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.title('PPO Training Progress on Farm Pest Control Environment')
    plt.grid(True)
    
    # Save figure
    plt.savefig(f"{log_dir}/learning_curve.png")
    plt.show()

# Plot the training results
plot_results()

# Function to evaluate the trained model
def evaluate_model(model, env, num_episodes=5, render=True):
    """Evaluate the trained model."""
    episode_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        while not done:
            # Get predicted action
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, rewards, dones, infos = env.step(action)
            
            # Update tracking variables
            episode_reward += rewards[0]
            done = dones[0]
            step += 1
            
            # Print step info if noteworthy
            if infos[0].get('pest_eliminated', False):
                print(f"Step {step}: Eliminated pest! Remaining: {infos[0].get('remaining_pests', 0)}")
            
            # Render the environment
            if render:
                env.render()
                time.sleep(0.05)
                
        # Episode complete
        is_success = infos[0].get('remaining_pests', 0) == 0
        if is_success:
            success_count += 1
            
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1} - Reward: {episode_reward:.2f}, Success: {is_success}")
        
    # Print summary
    print("\n--- Evaluation Results ---")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Success Rate: {success_count/num_episodes:.2%}")
    
    return episode_rewards, success_count/num_episodes

# Evaluate the trained model
print("\nEvaluating the trained model...")
# Create a fresh environment for evaluation
eval_env = DummyVecEnv([make_env(0)])
eval_env = VecNormalize.load(f"{log_dir}/vec_normalize.pkl", eval_env)
eval_env.training = False  # Don't update normalization statistics
eval_env.norm_reward = False  # Don't normalize rewards during evaluation

# Load the trained model
trained_model = PPO.load(f"{model_dir}/final_model")

# Run evaluation
rewards, success_rate = evaluate_model(trained_model, eval_env, num_episodes=5, render=True)

print("\nTraining and evaluation complete!")