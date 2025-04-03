import time
import numpy as np
from stable_baselines3 import DQN, PPO
from environment.custom_env import FarmPestControl  # Ensure correct import

# Load models
dqn_model = DQN.load("dqn_farm_pest_models/final_model")
ppo_model = PPO.load("ppo_farm_pest_models/final_model")

# Number of test episodes
num_episodes = 5  
max_steps = 50

def run_model(model, model_name):
    """ Runs a trained model for multiple episodes and evaluates its performance. """
    env = FarmPestControl(grid_size=(10, 10), num_pests=5, num_obstacles=10)
    total_rewards = []

    for episode in range(num_episodes):
        vec_env = model.get_env()
        obs = vec_env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            env.render()
            episode_reward += reward
            step_count += 1
            time.sleep(0.2)  # Delay for visualization
        if step_count >= max_steps:
            print("Warning: Episode did not terminate naturally.")

        total_rewards.append(episode_reward)
        print(f"{model_name} - Episode {episode + 1}: Reward = {episode_reward}, Steps = {step_count}")

    avg_reward = np.mean(total_rewards)
    print(f"\n{model_name} - Average Reward over {num_episodes} episodes: {avg_reward}")
    env.close()
    return avg_reward

# # Run and compare both models
print("\nTesting DQN Model:")
dqn_avg_reward = run_model(dqn_model, "DQN")

print("\nTesting PPO Model:")
ppo_avg_reward = run_model(ppo_model, "PPO")

# Compare performance
if dqn_avg_reward > ppo_avg_reward:
    print("\nDQN performed better overall!")
elif ppo_avg_reward > dqn_avg_reward:
    print("\nPPO performed better overall!")
else:
    print("\nBoth models performed equally well.")
