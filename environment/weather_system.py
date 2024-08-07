import numpy as np


class WeatherSystem:
    def __init__(self, env_width, env_height, weather_change_rate):
        self.env_width = env_width
        self.env_height = env_height
        self.weather_change_rate = weather_change_rate
        self.weather_map = np.zeros((env_width, env_height))
        self.weather_types = ["clear", "cloudy", "rainy", "stormy"]
        self.current_weather = "clear"
        self.update_interval = int(1 / weather_change_rate)
        self.time_since_last_update = 0
        self.wind_speed = np.zeros(2)
        self.wind_direction = 0
        self.temperature = 20  # 初始温度设为20°C
        self.humidity = 50  # 初始湿度设为50%

    def update(self):
        self.time_since_last_update += 1
        if self.time_since_last_update >= self.update_interval:
            self.generate_new_weather()
            self.update_wind()
            self.update_temperature()
            self.update_humidity()
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

    def update_wind(self):
        self.wind_direction += np.random.uniform(-0.1, 0.1)
        self.wind_speed[0] = np.cos(self.wind_direction) * np.random.uniform(0, 5)
        self.wind_speed[1] = np.sin(self.wind_direction) * np.random.uniform(0, 5)

    def update_temperature(self):
        if self.current_weather == "clear":
            self.temperature += np.random.uniform(-1, 1)
        elif self.current_weather == "cloudy":
            self.temperature += np.random.uniform(-0.5, 0.5)
        elif self.current_weather == "rainy":
            self.temperature -= np.random.uniform(0, 1)
        else:  # stormy
            self.temperature -= np.random.uniform(1, 2)

        self.temperature = np.clip(self.temperature, -10, 40)  # 将温度限制在-10°C到40°C之间

    def update_humidity(self):
        if self.current_weather == "clear":
            self.humidity -= np.random.uniform(0, 2)
        elif self.current_weather == "cloudy":
            self.humidity += np.random.uniform(0, 2)
        elif self.current_weather == "rainy":
            self.humidity += np.random.uniform(2, 5)
        else:  # stormy
            self.humidity += np.random.uniform(5, 10)

        self.humidity = np.clip(self.humidity, 0, 100)  # 将湿度限制在0%到100%之间

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

    def get_wind_effect(self, agent_type):
        if agent_type in ["uav", "satellite"]:
            return self.wind_speed
        return np.zeros(2)

    def get_temperature_effect(self, agent_type):
        # 温度对不同类型的智能体可能有不同的影响
        # 这里我们假设温度对所有智能体有相同的影响
        return max(0, 1 - abs(self.temperature - 20) / 30)

    def get_humidity_effect(self, agent_type):
        # 湿度对不同类型的智能体可能有不同的影响
        # 这里我们假设湿度只对地面智能体有影响
        if agent_type in ["uav", "satellite"]:
            return 1.0
        return max(0, 1 - abs(self.humidity - 50) / 100)

    def reset(self):
        self.weather_map.fill(0)
        self.current_weather = "clear"
        self.time_since_last_update = 0
        self.wind_speed = np.zeros(2)
        self.wind_direction = 0
        self.temperature = 20
        self.humidity = 50

    def get_state(self):
        return np.array([
            self.weather_types.index(self.current_weather) / len(self.weather_types),
            self.wind_speed[0] / 5,
            self.wind_speed[1] / 5,
            (self.temperature + 10) / 50,  # 归一化温度
            self.humidity / 100
        ])

    def get_state_dim(self):
        return 5  # 天气类型、风速(x,y)、温度、湿度