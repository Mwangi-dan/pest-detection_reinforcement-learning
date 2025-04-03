import os
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
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
log_dir = "dqn_farm_pest_logs/"
model_dir = "dqn_farm_pest_models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Function to create the environment
def make_env(rank, seed=0):
    def _init():
        env = FarmPestControl(grid_size=(10, 10), num_pests=5, num_obstacles=10, max_steps_per_episode=200)
        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _init

# Create the vectorized environment
env = DummyVecEnv([make_env(i) for i in range(1)])

# Wrap with VecNormalize for observation normalization
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

# Define hyperparameters - tuned for this specific environment
hyperparams = {
    "learning_rate": 5e-4,        # Learning rate - somewhat smaller for stable learning
    "buffer_size": 50000,         # Replay buffer size - moderate size for this simple environment
    "learning_starts": 1000,      # Initial steps before learning starts - allow some exploration first
    "batch_size": 64,             # Batch size - standard for DQN
    "tau": 0.1,                   # Soft update coefficient - moderately frequent updates
    "gamma": 0.99,                # Discount factor - standard for most RL tasks
    "train_freq": 4,              # Update the model every 4 steps - balance between stability and speed
    "gradient_steps": 1,          # How many gradient steps to do after each rollout
    "target_update_interval": 200, # How often to update target network - balance stability and adaptation speed
    "exploration_fraction": 0.2,  # Exploration rate decreases over this fraction of total timesteps
    "exploration_initial_eps": 1.0, # Start with 100% random actions
    "exploration_final_eps": 0.05, # End with 5% random actions
    "max_grad_norm": 10,          # Max norm for gradient clipping - helps with stability
}

# Create the policy (MlpPolicy is appropriate for this feature vector observation)
policy_kwargs = dict(
    net_arch=[128, 128]  # Two hidden layers with 128 neurons each
)

# Create the DQN agent
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=policy_kwargs,
    **hyperparams
)

# Setup callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=model_dir,
    name_prefix="dqn_farm_pest"
)

# Total timesteps for training
total_timesteps = 50000  # This can be adjusted based on convergence

# Train the agent
print("Starting training...")
model.learn(
    total_timesteps=total_timesteps,
    callback=checkpoint_callback,
    progress_bar=True,
)

# Save the final model
model.save(f"{model_dir}/final_model")
print(f"Training complete. Final model saved to {model_dir}/final_model")

# Function to test the trained model
def test_model(model, env, num_episodes=10, render=True):
    """Test the trained model."""
    obs = env.reset()
    
    total_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        episode_reward = 0
        done = False
        info = {}
        obs = env.reset()
        
        print(f"Episode {episode+1}/{num_episodes}")
        step = 0
        
        while not done:
            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take the action
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            
            # Render if specified
            if render:
                env.render()
                time.sleep(0.05)  # Small delay to view the agent's behavior
            
            step += 1
        
        # Episode completed
        is_success = info[0].get('remaining_pests', 0) == 0
        if is_success:
            success_count += 1
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward:.2f}, Success: {is_success}")
        
    # Print summary
    print(f"\nTesting Summary:")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Success rate: {success_count/num_episodes:.2%}")
    
    return total_rewards, success_count/num_episodes

# Plot training progress
def plot_results():
    """Plot the training progress."""
    plt.figure(figsize=(10, 6))
    
    # Load results
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    
    # Plot learning curve
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.title('DQN Training Progress on Farm Pest Control Environment')
    plt.grid(True)
    
    # Save figure
    plt.savefig(f"{log_dir}/learning_curve.png")
    plt.close()

# Plot the training results
plot_results()

# Optional: Test the trained model
print("\nTesting the trained model...")
env = DummyVecEnv([make_env(0)])
env = VecNormalize.load(f"{log_dir}/vec_normalize.pkl", env)
env.training = False  # Don't update the normalization statistics during testing
env.norm_reward = False  # Don't normalize rewards during testing

model = DQN.load(f"{model_dir}/final_model")
rewards, success_rate = test_model(model, env, num_episodes=5, render=True)

print("\nTraining and evaluation complete!")