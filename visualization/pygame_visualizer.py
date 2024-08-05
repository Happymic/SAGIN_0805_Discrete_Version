import pygame
import numpy as np
from shapely import Point, Polygon


class PygameVisualizer:
    def __init__(self, env, config):
        pygame.init()
        self.env = env
        self.config = config
        self.screen_width = config['screen_width']
        self.screen_height = config['screen_height']
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("SAGIN Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.colors = {
            "background": (240, 240, 240),
            "obstacle": (100, 100, 100),
            "poi": (0, 200, 0),
            "disaster": (200, 0, 0),
            "signal_detector": (0, 0, 255),
            "transport_vehicle": (255, 165, 0),
            "rescue_vehicle": (255, 0, 255),
            "uav": (0, 255, 255),
            "satellite": (128, 128, 128),
            "fixed_station": (165, 42, 42),
            "text": (0, 0, 0),
            "task_pending": (255, 255, 0),
            "task_completed": (0, 255, 0)
        }

    def update(self, env, episode, step, episode_reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill(self.colors["background"])
        self.draw_world(env)
        self.draw_agents(env)
        self.draw_tasks(env)
        self.draw_info(episode, step, episode_reward)

        pygame.display.flip()
        self.clock.tick(self.config['fps'])
        return True

    def draw_world(self, env):
        scale_x = self.screen_width / env.world.width
        scale_y = self.screen_height / env.world.height

        for obstacle in env.world.obstacles:
            if obstacle["type"] == "circle":
                x, y = int(obstacle["center"][0] * scale_x), int(obstacle["center"][1] * scale_y)
                radius = int(obstacle["radius"] * min(scale_x, scale_y))
                pygame.draw.circle(self.screen, self.colors["obstacle"], (x, y), radius)
            elif obstacle["type"] == "polygon":
                points = [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in obstacle["points"]]
                pygame.draw.polygon(self.screen, self.colors["obstacle"], points)

        for poi in env.world.pois:
            if isinstance(poi, tuple) and len(poi) == 2:
                x, y = int(poi[0] * scale_x), int(poi[1] * scale_y)
                pygame.draw.circle(self.screen, self.colors["poi"], (x, y), 5)
            else:
                print(f"Warning: Unexpected POI format: {poi}")
        for area in env.world.disaster_areas:
            if area["type"] == "circle":
                x, y = int(area["center"][0] * scale_x), int(area["center"][1] * scale_y)
                radius = int(area["radius"] * min(scale_x, scale_y))
                pygame.draw.circle(self.screen, self.colors["disaster"], (x, y), radius, 2)
            elif area["type"] == "polygon":
                points = [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in area["points"]]
                pygame.draw.polygon(self.screen, self.colors["disaster"], points, 2)

    def draw_agents(self, env):
        scale_x = self.screen_width / env.world.width
        scale_y = self.screen_height / env.world.height

        for agent in env.agents:
            x, y = int(agent.position[0] * scale_x), int(agent.position[1] * scale_y)
            pygame.draw.circle(self.screen, self.colors[agent.type], (x, y), 5)
            self.draw_text(agent.type[:2].upper(), (x - 10, y - 20), size=12)

    def draw_tasks(self, env):
        scale_x = self.screen_width / env.world.width
        scale_y = self.screen_height / env.world.height

        for task in env.tasks:
            start_x, start_y = int(task["start"][0] * scale_x), int(task["start"][1] * scale_y)
            end_x, end_y = int(task["end"][0] * scale_x), int(task["end"][1] * scale_y)
            color = self.colors["task_pending"] if task["status"] == "pending" else self.colors["task_completed"]
            pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 2)
            self.draw_text(f"T{task['id']}", (start_x - 10, start_y - 20), size=12)

    def draw_info(self, episode, step, episode_reward):
        info_text = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Reward: {episode_reward:.2f}",
            f"Time: {self.env.time:.1f}",
            f"Tasks: {len(self.env.tasks)}",
            f"Completed: {sum(1 for task in self.env.tasks if task['status'] == 'completed')}"
        ]
        for i, text in enumerate(info_text):
            self.draw_text(text, (10, 10 + i * 30))

    def draw_text(self, text, position, size=24, color=None):
        if color is None:
            color = self.colors["text"]
        font = pygame.font.Font(None, size)
        text_surface = font.render(str(text), True, color)
        self.screen.blit(text_surface, position)

    def should_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        return False

    def close(self):
        pygame.quit()