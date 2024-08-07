import pygame
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PygameVisualizer:
    def __init__(self, env, config):
        pygame.init()
        self.env = env
        self.config = config
        self.width = 1000
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("SAGIN Simulation")
        self.clock = pygame.time.Clock()

        self.map_width = int(self.width * 0.7)
        self.map_height = self.height
        self.sidebar_width = self.width - self.map_width

        self.map_surface = pygame.Surface((self.map_width, self.map_height))
        self.sidebar_surface = pygame.Surface((self.sidebar_width, self.height))

        self.font = pygame.font.Font(None, 24)
        self.colors = self.initialize_colors()

    def initialize_colors(self):
        return {
            'background': (240, 240, 240),
            'obstacle': (100, 100, 100),
            'agent': (0, 0, 255),
            'task': (255, 0, 0),
            'text': (0, 0, 0),
        }

    def update(self, env, episode, step, episode_reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(self.colors['background'])
        self.draw_map()
        self.draw_sidebar(episode, step, episode_reward)

        self.screen.blit(self.map_surface, (0, 0))
        self.screen.blit(self.sidebar_surface, (self.map_width, 0))

        pygame.display.flip()
        self.clock.tick(self.config['fps'])
        return True

    def draw_map(self):
        self.map_surface.fill(self.colors['background'])
        self.draw_obstacles()
        self.draw_agents()
        self.draw_tasks()

    def draw_obstacles(self):
        for obstacle in self.env.world.obstacles:
            if obstacle['type'] == 'rectangle':
                pos = self.world_to_screen(obstacle['center'])
                width = int(obstacle['width'] * self.map_width / self.env.world.width)
                height = int(obstacle['height'] * self.map_height / self.env.world.height)
                pygame.draw.rect(self.map_surface, self.colors['obstacle'],
                                 (pos[0] - width // 2, pos[1] - height // 2, width, height))

    def draw_agents(self):
        for agent in self.env.agents:
            pos = self.world_to_screen(agent.position)
            pygame.draw.circle(self.map_surface, self.colors['agent'], pos, 5)
            self.draw_text(self.map_surface, agent.id, (pos[0] + 10, pos[1] - 10))

    def draw_tasks(self):
        for task in self.env.tasks:
            pos = self.world_to_screen(task.get_current_target())
            pygame.draw.circle(self.map_surface, self.colors['task'], pos, 5)
            self.draw_text(self.map_surface, f"T{task.id}", (pos[0] + 10, pos[1] - 10))

    def draw_sidebar(self, episode, step, episode_reward):
        self.sidebar_surface.fill(self.colors['background'])
        info_texts = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Reward: {episode_reward:.2f}",
            f"Time: {self.env.time:.1f}",
            f"Tasks: {len(self.env.tasks)}",
            f"Completed: {sum(task.is_completed() for task in self.env.tasks)}",
        ]
        for i, text in enumerate(info_texts):
            self.draw_text(self.sidebar_surface, text, (10, 10 + i * 30))

        self.draw_task_info()
        self.draw_agent_info()

    def draw_task_info(self):
        y_offset = 200
        self.draw_text(self.sidebar_surface, "Task Information:", (10, y_offset))
        y_offset += 30
        for task in self.env.tasks:
            text = f"T{task.id}: {task.type} - Progress: {task.progress:.0f}%"
            self.draw_text(self.sidebar_surface, text, (10, y_offset))
            y_offset += 25

    def draw_agent_info(self):
        y_offset = 400
        self.draw_text(self.sidebar_surface, "Agent Information:", (10, y_offset))
        y_offset += 30
        for agent in self.env.agents:
            text = f"{agent.id}: E={agent.energy:.0f}, T={agent.current_task.id if agent.current_task else 'None'}"
            self.draw_text(self.sidebar_surface, text, (10, y_offset))
            y_offset += 25

    def world_to_screen(self, position):
        x = int(position[0] * self.map_width / self.env.world.width)
        y = int(position[1] * self.map_height / self.env.world.height)
        return (x, y)

    def draw_text(self, surface, text, position, color=(0, 0, 0)):
        text_surface = self.font.render(str(text), True, color)
        surface.blit(text_surface, position)

    def close(self):
        pygame.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            logger.error(f"An error occurred in PygameVisualizer: {exc_type}, {exc_val}")
            return False
        return True