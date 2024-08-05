import numpy as np
from abc import ABC, abstractmethod


class Sensor(ABC):
    def __init__(self, range, accuracy, failure_rate):
        self.range = range
        self.accuracy = accuracy
        self.failure_rate = failure_rate
        self.is_functioning = True

    @abstractmethod
    def sense(self, agent, environment):
        pass

    def check_failure(self):
        if np.random.random() < self.failure_rate:
            self.is_functioning = False
        else:
            self.is_functioning = True

    def add_noise(self, data):
        return data + np.random.normal(0, 1 - self.accuracy, data.shape)


class Camera(Sensor):
    def __init__(self, range=50, accuracy=0.9, failure_rate=0.01, resolution=(640, 480)):
        super().__init__(range, accuracy, failure_rate)
        self.resolution = resolution

    def sense(self, agent, environment):
        self.check_failure()
        if not self.is_functioning:
            return None

        objects = environment.get_objects_in_range(agent.position, self.range)
        image = np.zeros(self.resolution + (3,))  # RGB image

        for obj in objects:
            if np.random.random() < self.accuracy:
                color = self.get_object_color(obj)
                position = self.world_to_image(obj.position - agent.position)
                self.draw_object(image, position, color)

        return self.add_noise(image)

    def world_to_image(self, position):
        x = int((position[0] + self.range) / (2 * self.range) * self.resolution[0])
        y = int((position[1] + self.range) / (2 * self.range) * self.resolution[1])
        return (x, y)

    def draw_object(self, image, position, color):
        x, y = position
        image[max(0, y - 2):min(self.resolution[1], y + 3),
        max(0, x - 2):min(self.resolution[0], x + 3)] = color

    def get_object_color(self, obj):
        # Define colors for different object types
        colors = {
            'obstacle': [128, 128, 128],
            'agent': [0, 255, 0],
            'task': [255, 0, 0]
        }
        return colors.get(obj.type, [255, 255, 255])


class Lidar(Sensor):
    def __init__(self, range=100, accuracy=0.95, failure_rate=0.005, angular_resolution=1):
        super().__init__(range, accuracy, failure_rate)
        self.angular_resolution = angular_resolution

    def sense(self, agent, environment):
        self.check_failure()
        if not self.is_functioning:
            return None

        num_rays = int(360 / self.angular_resolution)
        distances = np.full(num_rays, self.range)

        for i in range(num_rays):
            angle = i * self.angular_resolution * np.pi / 180
            direction = np.array([np.cos(angle), np.sin(angle)])
            ray_end = agent.position + direction * self.range

            for obj in environment.get_objects_in_range(agent.position, self.range):
                intersection = self.ray_intersection(agent.position, ray_end, obj)
                if intersection is not None:
                    distance = np.linalg.norm(intersection - agent.position)
                    distances[i] = min(distances[i], distance)

        return self.add_noise(distances)

    def ray_intersection(self, start, end, obj):
        # Implement ray-object intersection
        # This will depend on the object types in your environment
        pass


class Radar(Sensor):
    def __init__(self, range=200, accuracy=0.8, failure_rate=0.02, min_detection_size=1):
        super().__init__(range, accuracy, failure_rate)
        self.min_detection_size = min_detection_size

    def sense(self, agent, environment):
        self.check_failure()
        if not self.is_functioning:
            return None

        detected_objects = []

        for obj in environment.get_objects_in_range(agent.position, self.range):
            if obj.size >= self.min_detection_size and np.random.random() < self.accuracy:
                relative_position = obj.position - agent.position
                relative_velocity = obj.velocity - agent.velocity
                detected_objects.append({
                    'position': self.add_noise(relative_position),
                    'velocity': self.add_noise(relative_velocity),
                    'size': obj.size
                })

        return detected_objects


class GPS(Sensor):
    def __init__(self, accuracy=0.95, failure_rate=0.001):
        super().__init__(range=float('inf'), accuracy=accuracy, failure_rate=failure_rate)

    def sense(self, agent, environment):
        self.check_failure()
        if not self.is_functioning:
            return None

        true_position = agent.position
        error = np.random.normal(0, 1 - self.accuracy, 2)
        return true_position + error