import pygame
import numpy as np

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
            "grid": (200, 200, 200),
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
        self.cell_size = min(self.screen_width // env.grid_world.width,
                             self.screen_height // env.grid_world.height)
        self.grid_offset_x = (self.screen_width - self.cell_size * env.grid_world.width) // 2
        self.grid_offset_y = (self.screen_height - self.cell_size * env.grid_world.height) // 2
        self.episode = 0
        self.step = 0

    def draw_grid(self):
        for x in range(self.env.grid_world.width + 1):
            start_pos = (self.grid_offset_x + x * self.cell_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + x * self.cell_size, self.grid_offset_y + self.env.grid_world.height * self.cell_size)
            pygame.draw.line(self.screen, self.colors["grid"], start_pos, end_pos)
        for y in range(self.env.grid_world.height + 1):
            start_pos = (self.grid_offset_x, self.grid_offset_y + y * self.cell_size)
            end_pos = (self.grid_offset_x + self.env.grid_world.width * self.cell_size, self.grid_offset_y + y * self.cell_size)
            pygame.draw.line(self.screen, self.colors["grid"], start_pos, end_pos)

    def draw_cell(self, x, y, color):
        pygame.draw.rect(self.screen, color,
                         (self.grid_offset_x + x * self.cell_size,
                          self.grid_offset_y + y * self.cell_size,
                          self.cell_size, self.cell_size))

    def draw_agents(self):
        for agent in self.env.agents:
            x, y = agent.position
            center = (int(self.grid_offset_x + (x + 0.5) * self.cell_size),
                      int(self.grid_offset_y + (y + 0.5) * self.cell_size))
            pygame.draw.circle(self.screen, self.colors[agent.type], center, int(self.cell_size * 0.4))
            self.draw_text(agent.type[:2].upper(), (center[0] - 10, center[1] - 10), size=18)

    def draw_tasks(self):
        for i, task in enumerate(self.env.tasks):
            start_x, start_y = task["start"]
            end_x, end_y = task["end"]
            start_pos = (int(self.grid_offset_x + (start_x + 0.5) * self.cell_size),
                         int(self.grid_offset_y + (start_y + 0.5) * self.cell_size))
            end_pos = (int(self.grid_offset_x + (end_x + 0.5) * self.cell_size),
                       int(self.grid_offset_y + (end_y + 0.5) * self.cell_size))
            color = self.colors["task_pending"] if task["status"] == "pending" else self.colors["task_completed"]
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
            self.draw_text(f"T{i}", (start_pos[0] - 10, start_pos[1] - 20), size=18)

    def draw_info(self):
        info_text = [
            f"Episode: {self.episode}",
            f"Step: {self.step}",
            f"Time: {self.env.time}",
            f"Tasks: {len(self.env.tasks)}",
            f"Completed: {sum(1 for task in self.env.tasks if task['status'] == 'completed')}"
        ]
        for i, text in enumerate(info_text):
            self.draw_text(text, (10, 10 + i * 30))

    def draw_text(self, text, position, size=24, color=None):
        if color is None:
            color = self.colors["text"]
        font = pygame.font.Font(None, size)
        try:
            text_surface = font.render(str(text), True, color)
            self.screen.blit(text_surface, position)
        except Exception as e:
            print(f"Error rendering text: {text}, Error: {e}")

    def update(self, episode, step):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill(self.colors["background"])
        self.draw_grid()

        for x, y in self.env.grid_world.obstacles:
            self.draw_cell(x, y, self.colors["obstacle"])
        for x, y in self.env.grid_world.pois:
            self.draw_cell(x, y, self.colors["poi"])
        for x, y in self.env.grid_world.disaster_areas:
            self.draw_cell(x, y, self.colors["disaster"])

        self.draw_agents()
        self.draw_tasks()
        self.draw_info(episode, step)

        pygame.display.flip()
        self.clock.tick(self.config['fps'])
        return True

    def draw_info(self, episode, step):
        info_text = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Time: {self.env.time}",
            f"Tasks: {len(self.env.tasks)}",
            f"Completed: {sum(1 for task in self.env.tasks if task['status'] == 'completed')}"
        ]
        for i, text in enumerate(info_text):
            self.draw_text(text, (10, 10 + i * 30))

    def close(self):
        pygame.quit()