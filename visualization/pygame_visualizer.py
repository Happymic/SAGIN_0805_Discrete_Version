import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


class PygameVisualizer:
    def __init__(self, env, config):
        pygame.init()
        self.env = env
        self.config = config
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("SAGIN Simulation")
        self.clock = pygame.time.Clock()

        # 设置主地图和侧边栏的尺寸
        self.map_width = int(self.width * 0.7)
        self.map_height = self.height
        self.sidebar_width = self.width - self.map_width
        self.sidebar_height = self.height

        # 创建surface
        self.map_surface = pygame.Surface((self.map_width, self.map_height))
        self.sidebar_surface = pygame.Surface((self.sidebar_width, self.sidebar_height))
        self.minimap_surface = pygame.Surface((self.sidebar_width, self.sidebar_width))

        self.fonts = {
            'small': pygame.font.Font(None, 18),
            'medium': pygame.font.Font(None, 24),
            'large': pygame.font.Font(None, 32)
        }

        self.colors = self.initialize_colors()
        self.selected_agent = None
        self.paused = False
        self.simulation_speed = 1.0
        self.show_sensor_range = False

        # 性能图表
        self.performance_data = {'rewards': [], 'task_completion': []}

    def initialize_colors(self):
        return {
            'background': (240, 240, 240),
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
            'sidebar_bg': (220, 220, 220),
            'button': (150, 150, 150),
            'button_hover': (180, 180, 180),
            'sensor_range': (200, 200, 200, 100)  # 半透明灰色
        }

    def update(self, env, episode, step, episode_reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self.handle_input(event)

        self.screen.fill(self.colors['background'])
        self.update_map()
        self.update_sidebar(episode, step, episode_reward)
        self.update_minimap()
        self.update_performance_graph()

        self.screen.blit(self.map_surface, (0, 0))
        self.screen.blit(self.sidebar_surface, (self.map_width, 0))
        self.screen.blit(self.minimap_surface, (self.map_width, self.height - self.sidebar_width))

        pygame.display.flip()
        self.clock.tick(self.config['fps'])
        return True

    def update_map(self):
        self.map_surface.fill(self.colors['background'])
        self.draw_obstacles()
        self.draw_agents()
        self.draw_tasks()
        if self.show_sensor_range:
            self.draw_sensor_ranges()

    def draw_obstacles(self):
        for obstacle in self.env.world.obstacles:
            if obstacle['type'] == 'rectangle':
                pos = self.world_to_screen(obstacle['center'])
                width = int(obstacle['width'] * self.map_width / self.env.world.width)
                height = int(obstacle['height'] * self.map_height / self.env.world.height)
                pygame.draw.rect(self.map_surface, self.colors['obstacle'],
                                 (pos[0] - width / 2, pos[1] - height / 2, width, height))

    def draw_agents(self):
        for agent in self.env.agents:
            color = self.colors['agent_colors'].get(type(agent).__name__, (128, 128, 128))
            pos = self.world_to_screen(agent.position)
            pygame.draw.circle(self.map_surface, color, pos, 5)
            self.draw_text(self.map_surface, agent.id, (pos[0] + 10, pos[1] - 10), self.fonts['small'])

            if agent.current_task:
                target = self.world_to_screen(agent.current_task.get_current_target())
                pygame.draw.line(self.map_surface, color, pos, target, 1)

    def draw_tasks(self):
        for task in self.env.tasks:
            if not task.is_completed():
                pos = self.world_to_screen(task.get_current_target())
                pygame.draw.polygon(self.map_surface, self.colors['task'],
                                    [(pos[0], pos[1] - 5), (pos[0] - 5, pos[1] + 5), (pos[0] + 5, pos[1] + 5)])

    def draw_sensor_ranges(self):
        for agent in self.env.agents:
            pos = self.world_to_screen(agent.position)
            radius = int(agent.sensor_range * self.map_width / self.env.world.width)
            sensor_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(sensor_surface, self.colors['sensor_range'], (radius, radius), radius)
            self.map_surface.blit(sensor_surface, (pos[0] - radius, pos[1] - radius))

    def update_sidebar(self, episode, step, episode_reward):
        self.sidebar_surface.fill(self.colors['sidebar_bg'])
        info_texts = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Reward: {episode_reward:.2f}",
            f"Time: {self.env.time:.1f}",
            f"Weather: {self.env.weather_system.current_weather}",
            f"Tasks: {len(self.env.tasks)}",
            f"Completed: {sum(task.is_completed() for task in self.env.tasks)}"
        ]
        for i, text in enumerate(info_texts):
            self.draw_text(self.sidebar_surface, text, (10, 10 + i * 25), self.fonts['medium'])

        if self.selected_agent:
            self.draw_agent_details(self.selected_agent)

        self.draw_button(self.sidebar_surface, "Pause/Resume", (10, self.sidebar_height - 100), self.paused)
        self.draw_button(self.sidebar_surface, "Toggle Sensor Range", (10, self.sidebar_height - 150),
                         self.show_sensor_range)
        self.draw_speed_slider()

    def update_minimap(self):
        self.minimap_surface.fill(self.colors['background'])
        scale_x = self.minimap_surface.get_width() / self.env.world.width
        scale_y = self.minimap_surface.get_height() / self.env.world.height

        for obstacle in self.env.world.obstacles:
            if obstacle['type'] == 'rectangle':
                pos = (int(obstacle['center'][0] * scale_x), int(obstacle['center'][1] * scale_y))
                width = int(obstacle['width'] * scale_x)
                height = int(obstacle['height'] * scale_y)
                pygame.draw.rect(self.minimap_surface, self.colors['obstacle'],
                                 (pos[0] - width / 2, pos[1] - height / 2, width, height))

        for agent in self.env.agents:
            color = self.colors['agent_colors'].get(type(agent).__name__, (128, 128, 128))
            pos = (int(agent.position[0] * scale_x), int(agent.position[1] * scale_y))
            pygame.draw.circle(self.minimap_surface, color, pos, 2)

    def update_performance_graph(self):
        if len(self.performance_data['rewards']) > 100:
            self.performance_data['rewards'] = self.performance_data['rewards'][-100:]
            self.performance_data['task_completion'] = self.performance_data['task_completion'][-100:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4))
        ax1.plot(self.performance_data['rewards'])
        ax1.set_title('Average Reward')
        ax2.plot(self.performance_data['task_completion'])
        ax2.set_title('Task Completion Rate')

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        surf = pygame.image.fromstring(raw_data, size, "RGB")
        self.sidebar_surface.blit(surf, (10, self.sidebar_height - 400))

        plt.close(fig)

    def handle_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                x, y = event.pos
                if self.map_width <= x <= self.width and self.height - 100 <= y <= self.height - 50:
                    self.paused = not self.paused
                elif self.map_width <= x <= self.width and self.height - 150 <= y <= self.height - 100:
                    self.show_sensor_range = not self.show_sensor_range
                elif x < self.map_width:
                    self.selected_agent = self.get_clicked_agent(event.pos)
                elif self.map_width <= x <= self.width and self.height - 30 <= y <= self.height:
                    self.update_simulation_speed(x)

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
            f"Position: ({agent.position[0]:.2f}, {agent.position[1]:.2f}, {agent.position[2]:.2f})",
            f"Energy: {agent.energy:.2f}",
            f"Task: {agent.current_task.id if agent.current_task else 'None'}",
            f"Sensor status: {'OK' if agent.is_functioning else 'Failure'}"
        ]
        for i, text in enumerate(details):
            self.draw_text(self.sidebar_surface, text, (10, 200 + i * 25), self.fonts['small'])

    def world_to_screen(self, position):
        x = int(position[0] * self.map_width / self.env.world.width)
        y = int(position[1] * self.map_height / self.env.world.height)
        return (x, y)

    def draw_text(self, surface, text, position, font, color=(0, 0, 0)):
        text_surface = font.render(str(text), True, color)
        surface.blit(text_surface, position)

    def draw_button(self, surface, text, position, is_active):
        button_color = self.colors['button_hover'] if is_active else self.colors['button']
        pygame.draw.rect(surface, button_color, (*position, 180, 40))
        self.draw_text(surface, text, (position[0] + 10, position[1] + 10), self.fonts['small'])

    def draw_speed_slider(self):
        pygame.draw.rect(self.sidebar_surface, (150, 150, 150),
                         (10, self.sidebar_height - 30, self.sidebar_width - 20, 20))
        slider_pos = int(10 + (self.sidebar_width - 20) * (self.simulation_speed - 0.1) / 1.9)
        pygame.draw.rect(self.sidebar_surface, (100, 100, 200), (slider_pos - 5, self.sidebar_height - 35, 10, 30))

    def update_simulation_speed(self, x):
        speed_range = self.sidebar_width - 20
        speed = 0.1 + (x - self.map_width - 10) / speed_range * 1.9
        self.simulation_speed = max(0.1, min(2.0, speed))

    def update_performance_data(self, reward, task_completion_rate):
        self.performance_data['rewards'].append(reward)
        self.performance_data['task_completion'].append(task_completion_rate)

    def close(self):
        pygame.quit()