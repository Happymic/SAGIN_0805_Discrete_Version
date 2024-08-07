import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib

class ContinuousWorld:
    def __init__(self, width, height, num_obstacles, num_pois):
        self.width = width
        self.height = height
        self.obstacles = self.generate_obstacles(num_obstacles)
        self.pois = self.generate_pois(num_pois)
        self.charging_stations = self.generate_charging_stations()

    def generate_obstacles(self, num_obstacles):
        obstacles = []
        grid_size = 20  # 网格大小
        for i in range(0, self.width, grid_size):
            for j in range(0, self.height, grid_size):
                if np.random.random() < 0.05:  # 30% 概率生成障碍物
                    obstacle_size = np.random.uniform(5, 15)
                    center = (i + grid_size / 2, j + grid_size / 2)
                    height = np.random.uniform(5, 15)
                    obstacles.append({
                        "type": "rectangle",
                        "center": center,
                        "width": obstacle_size,
                        "height": obstacle_size,
                        "z_height": height
                    })
        return obstacles
    def is_valid_position(self, position, agent_type, agent_size, altitude):
        if not (0 <= position[0] < self.width and 0 <= position[1] < self.height):
            return False

        if agent_type in ["air", "space"]:
            if not (0 <= position[2] <= self.height):
                return False
        else:
            position = np.array([position[0], position[1], altitude])

        point = Point(position[:2])

        for obstacle in self.obstacles:
            if agent_type == "ground" or obstacle['z_height'] > position[2]:
                if obstacle['type'] == 'rectangle':
                    half_width = obstacle['width'] / 2
                    half_height = obstacle['height'] / 2
                    center = obstacle['center']
                    obstacle_polygon = Polygon([
                        (center[0] - half_width, center[1] - half_height),
                        (center[0] + half_width, center[1] - half_height),
                        (center[0] + half_width, center[1] + half_height),
                        (center[0] - half_width, center[1] + half_height)
                    ])
                    if obstacle_polygon.distance(point) <= agent_size:
                        return False

        return True

    def generate_pois(self, num_pois):
        return [Point(np.random.uniform(0, self.width), np.random.uniform(0, self.height)) for _ in range(num_pois)]

    def generate_charging_stations(self, num_stations=5):
        return [Point(np.random.uniform(0, self.width), np.random.uniform(0, self.height)) for _ in range(num_stations)]

    def get_random_valid_position(self, agent_type, agent_size):
        while True:
            if agent_type == "space":
                position = np.random.uniform(0, [self.width, self.height, self.height])
            elif agent_type == "air":
                position = np.random.uniform(0, [self.width, self.height,
                                                 100])  # Assuming max height of 100 for air agents
            else:
                position = np.random.uniform(0, [self.width, self.height, 0])

            altitude = position[2] if agent_type in ["air", "space"] else 0
            if self.is_valid_position(position, agent_type, agent_size, altitude):
                return position
    def get_nearest_poi(self, position):
        return min(self.pois, key=lambda poi: poi.distance(Point(position)))

    def get_nearest_charging_station(self, position):
        return min(self.charging_stations, key=lambda station: station.distance(Point(position)))

    def get_objects_in_range(self, position, range, agent_type, agent_altitude=0):
        objects = []
        point = Point(position[0], position[1])

        for obstacle in self.obstacles:
            if agent_type == "ground" or obstacle['height'] > agent_altitude:
                if obstacle['type'] == 'circle':
                    center = Point(obstacle['center'])
                    if point.distance(center) <= range + obstacle['radius']:
                        objects.append(obstacle)
                elif obstacle['type'] == 'polygon':
                    polygon = Polygon(obstacle['points'])
                    if polygon.distance(point) <= range:
                        objects.append(obstacle)

        objects.extend([poi for poi in self.pois if poi.distance(point) <= range])
        objects.extend([station for station in self.charging_stations if station.distance(point) <= range])

        return objects

    def get_state(self):
        # Return a flattened representation of the world state
        state = []
        for obstacle in self.obstacles:
            if obstacle['type'] == 'circle':
                state.extend([obstacle['center'][0], obstacle['center'][1], obstacle['radius'], obstacle['height']])
            else:  # polygon
                state.extend([coord for point in obstacle['points'] for coord in point] + [obstacle['height']])
        for poi in self.pois:
            state.extend([poi.x, poi.y])
        for station in self.charging_stations:
            state.extend([station.x, station.y])
        return np.array(state)

    def get_state_dim(self):
        return len(self.get_state())