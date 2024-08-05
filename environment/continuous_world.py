import numpy as np
from shapely.geometry import Point, Polygon


class ContinuousWorld:
    def __init__(self, width, height, num_obstacles, num_pois):
        self.width = width
        self.height = height
        self.obstacles = self.generate_obstacles(num_obstacles)
        self.pois = self.generate_pois(num_pois)
        self.charging_stations = self.generate_charging_stations()

    def generate_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            if np.random.random() < 0.7:  # 70% chance of circular obstacle
                center = (np.random.uniform(0, self.width), np.random.uniform(0, self.height))
                radius = np.random.uniform(1, 5)
                obstacles.append({"type": "circle", "center": center, "radius": radius})
            else:  # 30% chance of polygon obstacle
                points = [(np.random.uniform(0, self.width), np.random.uniform(0, self.height)) for _ in
                          range(np.random.randint(3, 6))]
                obstacles.append({"type": "polygon", "points": points})
        return obstacles

    def generate_pois(self, num_pois):
        return [Point(np.random.uniform(0, self.width), np.random.uniform(0, self.height)) for _ in range(num_pois)]

    def generate_charging_stations(self, num_stations=5):
        return [Point(np.random.uniform(0, self.width), np.random.uniform(0, self.height)) for _ in range(num_stations)]

    def is_valid_position(self, position):
        if not (0 <= position[0] < self.width and 0 <= position[1] < self.height):
            return False

        point = Point(position)
        for obstacle in self.obstacles:
            if obstacle["type"] == "circle":
                if point.distance(Point(obstacle["center"])) <= obstacle["radius"]:
                    return False
            elif obstacle["type"] == "polygon":
                if Polygon(obstacle["points"]).contains(point):
                    return False
        return True

    def get_nearest_poi(self, position):
        return min(self.pois, key=lambda poi: poi.distance(Point(position)))

    def get_nearest_charging_station(self, position):
        return min(self.charging_stations, key=lambda station: station.distance(Point(position)))

    def get_objects_in_range(self, position, range):
        objects = []
        point = Point(position)
        for obstacle in self.obstacles:
            if obstacle["type"] == "circle":
                if point.distance(Point(obstacle["center"])) <= range + obstacle["radius"]:
                    objects.append(obstacle)
            elif obstacle["type"] == "polygon":
                if Polygon(obstacle["points"]).distance(point) <= range:
                    objects.append(obstacle)

        objects.extend([poi for poi in self.pois if poi.distance(point) <= range])
        objects.extend([station for station in self.charging_stations if station.distance(point) <= range])

        return objects