import numpy as np

class WeatherSystem:
    def __init__(self, env_width, env_height):
        self.env_width = env_width
        self.env_height = env_height
        self.weather_map = np.zeros((env_width, env_height))
        self.weather_types = ["clear", "cloudy", "rainy", "stormy"]
        self.current_weather = "clear"
        self.update_interval = 100  # Update weather every 100 time steps
        self.time_since_last_update = 0

    def update(self):
        self.time_since_last_update += 1
        if self.time_since_last_update >= self.update_interval:
            self.generate_new_weather()
            self.time_since_last_update = 0

    def generate_new_weather(self):
        self.current_weather = np.random.choice(self.weather_types, p=[0.5, 0.3, 0.15, 0.05])
        if self.current_weather == "clear":
            self.weather_map.fill(0)
        elif self.current_weather == "cloudy":
            self.weather_map = np.random.uniform(0, 0.5, (self.env_width, self.env_height))
        elif self.current_weather == "rainy":
            self.weather_map = np.random.uniform(0.5, 0.8, (self.env_width, self.env_height))
        else:  # stormy
            self.weather_map = np.random.uniform(0.8, 1, (self.env_width, self.env_height))

    def get_weather_at_position(self, position):
        x, y = int(position[0]), int(position[1])
        return self.weather_map[x, y]

    def get_weather_effect(self, agent_type):
        if self.current_weather == "clear":
            return 1.0
        elif self.current_weather == "cloudy":
            return 0.9 if agent_type in ["uav", "satellite"] else 1.0
        elif self.current_weather == "rainy":
            return 0.7 if agent_type in ["uav", "satellite"] else 0.9
        else:  # stormy
            return 0.5 if agent_type in ["uav", "satellite"] else 0.7