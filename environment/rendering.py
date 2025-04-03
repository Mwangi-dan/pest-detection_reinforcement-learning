import pygame
from pygame.locals import *

def init_render(grid_size, cell_size):
    """Initializes Pygame for rendering."""
    pygame.init()
    screen = pygame.display.set_mode((grid_size[1] * cell_size, grid_size[0] * cell_size))
    font = pygame.font.Font(None, 18)
    return screen, font

def render(screen, font, grid_size, cell_size, agent_position, pest_positions, obstacle_positions):
    """Renders the current state of the environment."""

    screen.fill((30, 30, 30))  # Dark background

    # Draw Grid
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)  # Darker grid lines

    # Draw Agent
    agent_rect = pygame.Rect(agent_position[1] * cell_size, agent_position[0] * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, (255, 0, 0), agent_rect)  # Red agent
    agent_text = font.render("A", True, (255, 255, 255))
    screen.blit(agent_text, agent_rect.center)

    # Draw Pests
    for pest_pos in pest_positions:
        pest_rect = pygame.Rect(pest_pos[1] * cell_size, pest_pos[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (0, 255, 0), pest_rect)  # Green pests
        pest_text = font.render("P", True, (255, 255, 255))
        screen.blit(pest_text, pest_rect.center)

    # Draw Obstacles
    for obstacle_pos in obstacle_positions:
        obstacle_rect = pygame.Rect(obstacle_pos[1] * cell_size, obstacle_pos[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (100, 100, 100), obstacle_rect)  # Darker grey obstacles.
        obstacle_text = font.render("O", True, (255, 255, 255))
        screen.blit(obstacle_text, obstacle_rect.center)

    pygame.display.flip()

    # Handle events (especially quitting)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False  # Signal to stop the main loop
    return True  # Signal to continue the main loop

def close_render():
    """Closes Pygame."""
    pygame.quit()