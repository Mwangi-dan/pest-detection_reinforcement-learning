import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class FarmPestControl(gym.Env):
    """
    Enhanced Gym environment for pest control where:
    - Agent automatically eliminates pests when landing on them
    - No cap on obstacles hit
    - Focus on efficient pest elimination and navigation
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size=(10, 10), num_pests=5, num_obstacles=10, cell_size=30, max_steps_per_episode=150):
        super(FarmPestControl, self).__init__()
        self.grid_size = grid_size
        self.grid_height = grid_size[0]
        self.grid_width = grid_size[1]
        self.num_pests_start = num_pests
        self.num_obstacles = num_obstacles
        self.cell_size = cell_size
        self.max_steps_per_episode = max_steps_per_episode
        self._current_step = 0
        
        # Action space: 0:Up, 1:Down, 2:Left, 3:Right
        # Removed spray action since it's now automatic
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 1D Feature Vector
        # Features:
        # 1. Agent Row (normalized)
        # 2. Agent Col (normalized)
        # 3. Nearest Pest Delta Row (normalized)
        # 4. Nearest Pest Delta Col (normalized)
        # 5. Remaining Pests (normalized)
        num_features = 5
        self.observation_space = spaces.Box(
            low=-1.0,  # Relative coords can be negative
            high=1.0,
            shape=(num_features,),
            dtype=np.float32
        )
        
        # Pygame rendering state - Initialized Lazily
        self.screen = None
        self.font = None
        self.is_render_initialized = False
        
        # State variables (initialized in reset)
        self.agent_position = None
        self.pest_positions = None
        self.obstacle_positions = None
        self.remaining_pests = 0
        self.obstacles_hit_total = 0
        self.start_position = (0, 0)  # Define start position
        
    def _initialize_render(self):
        """Initializes Pygame rendering if not already done."""
        if not self.is_render_initialized:
            print("Initializing Pygame rendering...")
            try:
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.grid_width * self.cell_size, self.grid_height * self.cell_size)
                )
                pygame.display.set_caption('Farm Pest Control')
                self.font = pygame.font.SysFont(None, 24)
                self.is_render_initialized = True
                print("Pygame rendering initialized successfully.")
            except Exception as e:
                print(f"WARNING: Error initializing Pygame: {e}. Human rendering disabled.")
                self.is_render_initialized = False
                
    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.agent_position = self.start_position  # Agent starts at the defined start position
        self._current_step = 0
        self.obstacles_hit_total = 0
        
        # Create RNG if needed
        if not hasattr(self, 'np_random'):
            self.np_random = np.random.RandomState(seed)
            
        # Ensure Unique Pest Positions
        self.pest_positions = []
        while len(self.pest_positions) < self.num_pests_start:
            pest_pos = (self.np_random.integers(0, self.grid_height),
                        self.np_random.integers(0, self.grid_width))
            if pest_pos != self.agent_position and pest_pos not in self.pest_positions:
                self.pest_positions.append(pest_pos)
                
        # Ensure Unique Obstacle Positions
        self.obstacle_positions = []
        while len(self.obstacle_positions) < self.num_obstacles:
            obstacle_pos = (self.np_random.integers(0, self.grid_height),
                            self.np_random.integers(0, self.grid_width))
            if obstacle_pos != self.agent_position and \
               obstacle_pos not in self.pest_positions and \
               obstacle_pos not in self.obstacle_positions:
                self.obstacle_positions.append(obstacle_pos)
                
        self.remaining_pests = self.num_pests_start
        observation = self._get_observation()
        info = {}
        
        return observation, info
        
    def _get_observation(self):
        """Generates the 1D feature vector observation."""
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 1. Normalized Agent Position
        obs[0] = self.agent_position[0] / (self.grid_height - 1) if self.grid_height > 1 else 0
        obs[1] = self.agent_position[1] / (self.grid_width - 1) if self.grid_width > 1 else 0
        
        # 2. Normalized Relative Position to Nearest Pest
        nearest_pest_dist = float('inf')
        nearest_pest_rel_pos = (0, 0)  # Default if no pests
        
        if self.pest_positions:  # Check if there are pests left
            agent_r, agent_c = self.agent_position
            for pest_r, pest_c in self.pest_positions:
                # Using Euclidean distance
                dist_sq = (pest_r - agent_r)**2 + (pest_c - agent_c)**2
                if dist_sq < nearest_pest_dist:
                    nearest_pest_dist = dist_sq
                    # Calculate relative position
                    delta_r = pest_r - agent_r
                    delta_c = pest_c - agent_c
                    # Normalize relative position
                    norm_delta_r = delta_r / (self.grid_height - 1) if self.grid_height > 1 else 0
                    norm_delta_c = delta_c / (self.grid_width - 1) if self.grid_width > 1 else 0
                    nearest_pest_rel_pos = (norm_delta_r, norm_delta_c)
                    
        obs[2] = nearest_pest_rel_pos[0]
        obs[3] = nearest_pest_rel_pos[1]
        
        # 3. Normalized Remaining Pests
        obs[4] = (self.remaining_pests / self.num_pests_start) if self.num_pests_start > 0 else 0
        
        # Clip observation values
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        return obs.astype(np.float32)
        
    def _eliminate_pest(self):
        """Automatically eliminates a pest if the agent is on the same position."""
        eliminated_pest = False
        if self.agent_position in self.pest_positions:
            self.pest_positions.remove(self.agent_position)
            self.remaining_pests -= 1
            eliminated_pest = True
        return eliminated_pest
        
    def step(self, action):
        """Take a step in the environment."""
        original_position = self.agent_position
        self._current_step += 1
        
        # Process movement actions (0-3)
        intended_pos = list(self.agent_position)
        if action == 0:  # Up
            intended_pos[0] = max(0, self.agent_position[0] - 1)
        elif action == 1:  # Down
            intended_pos[0] = min(self.grid_height - 1, self.agent_position[0] + 1)
        elif action == 2:  # Left
            intended_pos[1] = max(0, self.agent_position[1] - 1)
        elif action == 3:  # Right
            intended_pos[1] = min(self.grid_width - 1, self.agent_position[1] + 1)
            
        intended_pos = tuple(intended_pos)
        hit_obstacle = False
        
        # Check for obstacles
        if intended_pos in self.obstacle_positions:
            self.agent_position = original_position  # Can't move into obstacle
            self.obstacles_hit_total += 1
            hit_obstacle = True
        else:
            self.agent_position = intended_pos
                
        # Automatic pest elimination when landing on a pest
        pest_eliminated = self._eliminate_pest()
        
        # Calculate reward
        reward = self._calculate_reward(original_position, hit_obstacle, pest_eliminated)
        
        # Check termination conditions
        terminated = self.remaining_pests == 0  # Episode ends when all pests eliminated
        truncated = self._current_step >= self.max_steps_per_episode
        
        observation = self._get_observation()
        
        info = {
            'pest_eliminated': pest_eliminated,
            'hit_obstacle': hit_obstacle,
            'remaining_pests': self.remaining_pests,
            'obstacles_hit': self.obstacles_hit_total,
            'steps': self._current_step
        }
        
        return observation, reward, terminated, truncated, info
        
    def _calculate_reward(self, original_position, hit_obstacle, pest_eliminated):
        """Calculate reward based on actions and outcomes."""
        reward = -0.01  # Small negative reward for each step (encourages efficiency)
        
        if pest_eliminated:
            reward += 1.0  # Strong positive reward for eliminating a pest
            
        if hit_obstacle:
            reward -= 0.2  # Negative reward for hitting an obstacle
            
        # Completion bonus - add a large bonus if all pests are eliminated
        if self.remaining_pests == 0:
            reward += 2.0
            
            # Additional bonus if returning to start after eliminating all pests
            if self.agent_position == self.start_position:
                reward += 3.0
                
        return reward
        
    def render(self, mode='human'):
        """Renders the environment."""
        if not self.is_render_initialized:
            self._initialize_render()
            
        if self.screen is not None and self.font is not None:
            # Clear the screen
            self.screen.fill((220, 255, 220))  # Light green background
            
            # Draw grid lines
            for x in range(0, self.grid_width * self.cell_size, self.cell_size):
                pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.grid_height * self.cell_size))
            for y in range(0, self.grid_height * self.cell_size, self.cell_size):
                pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.grid_width * self.cell_size, y))
                
            # Draw start position
            start_rect = pygame.Rect(
                self.start_position[1] * self.cell_size, 
                self.start_position[0] * self.cell_size,
                self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.screen, (100, 200, 100), start_rect)  # Light green
            
            # Draw obstacles
            for obs_pos in self.obstacle_positions:
                obs_rect = pygame.Rect(
                    obs_pos[1] * self.cell_size, 
                    obs_pos[0] * self.cell_size,
                    self.cell_size, self.cell_size
                )
                pygame.draw.rect(self.screen, (100, 100, 100), obs_rect)  # Gray
                
            # Draw pests
            for pest_pos in self.pest_positions:
                pest_rect = pygame.Rect(
                    pest_pos[1] * self.cell_size + self.cell_size // 4, 
                    pest_pos[0] * self.cell_size + self.cell_size // 4,
                    self.cell_size // 2, self.cell_size // 2
                )
                pygame.draw.rect(self.screen, (255, 0, 0), pest_rect)  # Red
                
            # Draw agent
            agent_rect = pygame.Rect(
                self.agent_position[1] * self.cell_size + self.cell_size // 4, 
                self.agent_position[0] * self.cell_size + self.cell_size // 4,
                self.cell_size // 2, self.cell_size // 2
            )
            pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)  # Blue
            
            # Draw info text
            info_text = f"Pests: {self.remaining_pests}/{self.num_pests_start} | Steps: {self._current_step} | Obstacles hit: {self.obstacles_hit_total}"
            text_surface = self.font.render(info_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10))
            
            pygame.display.flip()
            
        if mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen)
        return None
        
    def close(self):
        """Closes the environment."""
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.quit()
            self.is_render_initialized = False
            self.screen = None
            self.font = None

    
    def seed(self, seed=None):
        """Sets the seed for the environment."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

import time
if __name__ == "__main__":
    # Create the environment
    env = FarmPestControl(grid_size=(8, 8), num_pests=3, num_obstacles=5)

    # Prepare for training
    observation, info = env.reset()
    total_reward = 0
    done = False

    # Run a simple episode
    while not done:
        # Random action (in real training you'd use your RL algorithm here)
        action = env.action_space.sample()
        
        # Take a step
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Optional: render the environment
        env.render()
        time.sleep(0.1)
        # Check if episode is done
        done = terminated or truncated
        
        if info['pest_eliminated']:
            print(f"Pest eliminated! Remaining: {info['remaining_pests']}")
        if info['hit_obstacle']:
            print(f"Hit an obstacle! Total hits: {info['obstacles_hit']}")

    print(f"Episode finished with total reward: {total_reward:.2f}")
    env.close()
