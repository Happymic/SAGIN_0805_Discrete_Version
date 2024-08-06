import pygame
import numpy as np
from pygame import gfxdraw


class PygameVisualizer:
    def __init__(self, env, config):
        pygame.init()
        self.env = env
        self.config = config
        self.width = config['screen_width']
        self.height = config['screen_height']
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("SAGIN Simulation")
        self.clock = pygame.time.Clock()

        self.main_map_surface = pygame.Surface((int(self.width * 0.75), int(self.height * 0.8)))
        self.info_panel_surface = pygame.Surface((int(self.width * 0.25), int(self.height * 0.8)))
        self.control_panel_surface = pygame.Surface((self.width, int(self.height * 0.2)))
        self.mini_map_surface = pygame.Surface((int(self.width * 0.2), int(self.height * 0.2)))

        self.fonts = {
            'small': pygame.font.Font(None, 18),
            'medium': pygame.font.Font(None, 24),
            'large': pygame.font.Font(None, 32)
        }

        self.colors = self.initialize_colors()
        self.camera = {'x': 0, 'y': 0, 'zoom': 1}
        self.selected_agent = None
        self.visualization_mode = 'normal'

    def initialize_colors(self):
        return {
            'background': (240, 240, 240),
            'terrain': [(50, 100, 50), (100, 200, 100), (200, 230, 180)],
            'water': (100, 150, 255),
            'obstacle': (100, 100, 100),
            'agent_colors': {
                'SignalDetector': (255, 255, 0),
                'TransportVehicle': (0, 0, 255),
                'RescueVehicle': (255, 0, 0),
                'UAV': (0, 255, 0),
                'Satellite': (200, 200, 200),
                'FixedStation': (128, 0, 128)
            },
            'task': (255, 165, 0),
            'text': (0, 0, 0),
            'panel_bg': (220, 220, 220)
        }

    def update(self, env, episode, step, episode_reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self.handle_input(event)

        self.screen.fill(self.colors['background'])
        self.update_main_map()
        self.update_info_panel(episode, step, episode_reward)
        self.update_control_panel()
        self.update_mini_map()

        self.screen.blit(self.main_map_surface, (0, 0))
        self.screen.blit(self.info_panel_surface, (int(self.width * 0.75), 0))
        self.screen.blit(self.control_panel_surface, (0, int(self.height * 0.8)))
        self.screen.blit(self.mini_map_surface, (int(self.width * 0.8), int(self.height * 0.8)))

        pygame.display.flip()
        self.clock.tick(self.config['fps'])
        return True

    def update_main_map(self):
        self.main_map_surface.fill(self.colors['background'])
        self.draw_terrain()
        self.draw_obstacles()
        self.draw_agents()
        self.draw_tasks()

    def draw_terrain(self):
        terrain = self.env.terrain.elevation
        scale_x = self.main_map_surface.get_width() / terrain.shape[0]
        scale_y = self.main_map_surface.get_height() / terrain.shape[1]
        for x in range(terrain.shape[0]):
            for y in range(terrain.shape[1]):
                height = terrain[x, y]
                color = self.get_terrain_color(height)
                rect = pygame.Rect(int(x * scale_x), int(y * scale_y), int(scale_x) + 1, int(scale_y) + 1)
                pygame.draw.rect(self.main_map_surface, color, rect)

    def get_terrain_color(self, height):
        if height < 0.2:
            return self.colors['water']
        elif height < 0.5:
            return self.colors['terrain'][0]
        elif height < 0.8:
            return self.colors['terrain'][1]
        else:
            return self.colors['terrain'][2]

    def draw_obstacles(self):
        for obstacle in self.env.world.obstacles:
            if obstacle['type'] == 'circle':
                pygame.draw.circle(self.main_map_surface, self.colors['obstacle'],
                                   self.world_to_screen(obstacle['center']),
                                   int(obstacle['radius'] * self.main_map_surface.get_width() / self.env.world.width))
            elif obstacle['type'] == 'polygon':
                pygame.draw.polygon(self.main_map_surface, self.colors['obstacle'],
                                    [self.world_to_screen(point) for point in obstacle['points']])

    def draw_agents(self):
        for agent in self.env.agents:
            color = self.colors['agent_colors'].get(type(agent).__name__, (128, 128, 128))
            pos = self.world_to_screen(agent.position)
            pygame.draw.circle(self.main_map_surface, color, pos, 5)
            self.draw_text(self.main_map_surface, agent.id, (pos[0] + 10, pos[1] - 10), self.fonts['small'])

    def draw_tasks(self):
        for task in self.env.tasks:
            if not task.is_completed():
                pos = self.world_to_screen(task.get_current_target())
                pygame.draw.polygon(self.main_map_surface, self.colors['task'],
                                    [(pos[0], pos[1] - 5), (pos[0] - 5, pos[1] + 5), (pos[0] + 5, pos[1] + 5)])

    def update_info_panel(self, episode, step, episode_reward):
        self.info_panel_surface.fill(self.colors['panel_bg'])
        info_texts = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Reward: {episode_reward:.2f}",
            f"Time: {self.env.time:.1f}",
            f"Weather: {self.env.weather_system.current_weather}"
        ]
        for i, text in enumerate(info_texts):
            self.draw_text(self.info_panel_surface, text, (10, 10 + i * 25), self.fonts['medium'])

        if self.selected_agent:
            self.draw_agent_details(self.selected_agent)

    def update_control_panel(self):
        self.control_panel_surface.fill(self.colors['panel_bg'])
        self.draw_text(self.control_panel_surface, "Controls", (10, 10), self.fonts['large'])
        self.draw_text(self.control_panel_surface, "Click: Select Agent", (10, 50), self.fonts['medium'])
        self.draw_text(self.control_panel_surface, "Scroll: Zoom", (10, 80), self.fonts['medium'])

    def update_mini_map(self):
        self.mini_map_surface.fill(self.colors['background'])
        scale_x = self.mini_map_surface.get_width() / self.env.world.width
        scale_y = self.mini_map_surface.get_height() / self.env.world.height

        for agent in self.env.agents:
            color = self.colors['agent_colors'].get(type(agent).__name__, (128, 128, 128))
            pos = (int(agent.position[0] * scale_x), int(agent.position[1] * scale_y))
            pygame.draw.circle(self.mini_map_surface, color, pos, 2)

    def handle_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # 左键点击
                clicked_agent = self.get_clicked_agent(event.pos)
                if clicked_agent:
                    self.selected_agent = clicked_agent
            elif event.button == 4:  # 滚轮上滚
                self.camera['zoom'] = min(2.0, self.camera['zoom'] * 1.1)
            elif event.button == 5:  # 滚轮下滚
                self.camera['zoom'] = max(0.5, self.camera['zoom'] / 1.1)

    def get_clicked_agent(self, pos):
        for agent in self.env.agents:
            agent_pos = self.world_to_screen(agent.position)
            if ((pos[0] - agent_pos[0]) ** 2 + (pos[1] - agent_pos[1]) ** 2) ** 0.5 < 5:
                return agent
        return None

    def draw_agent_details(self, agent):
        details = [
            f"ID: {agent.id}",
            f"Type: {type(agent).__name__}",
            f"Position: ({agent.position[0]:.2f}, {agent.position[1]:.2f})",
            f"Energy: {agent.energy:.2f}"
        ]
        for i, text in enumerate(details):
            self.draw_text(self.info_panel_surface, text, (10, 150 + i * 25), self.fonts['small'])

    def world_to_screen(self, position):
        x = int((position[0] - self.camera['x']) * self.camera[
            'zoom'] * self.main_map_surface.get_width() / self.env.world.width)
        y = int((position[1] - self.camera['y']) * self.camera[
            'zoom'] * self.main_map_surface.get_height() / self.env.world.height)
        return (x, y)

    def draw_text(self, surface, text, position, font, color=(0, 0, 0)):
        text_surface = font.render(str(text), True, color)
        surface.blit(text_surface, position)

    def close(self):
        pygame.quit()