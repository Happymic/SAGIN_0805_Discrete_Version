import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

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
            logger.warning(f"Camera of agent {agent.id} is not functioning")
            return None

        objects = environment.get_objects_in_range(agent.position, self.range)
        logger.debug(f"Camera sensing {len(objects)} objects for agent {agent.id}")
        image = np.zeros(self.resolution + (3,))  # RGB image

        for obj in objects:
            if np.random.random() < self.accuracy:
                color = self.get_object_color(obj)
                if isinstance(obj, dict):
                    if 'center' in obj:
                        position = self.world_to_image(np.array(obj['center']) - agent.position)
                    elif 'position' in obj:
                        position = self.world_to_image(np.array(obj['position']) - agent.position)
                    else:
                        logger.warning(f"Object {obj} has no position information")
                        continue
                else:
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
        colors = {
            'obstacle': [128, 128, 128],
            'agent': [0, 255, 0],
            'task': [255, 0, 0]
        }
        if isinstance(obj, dict):
            return colors.get(obj.get('type', 'unknown'), [255, 255, 255])
        elif hasattr(obj, 'type'):
            return colors.get(obj.type, [255, 255, 255])
        else:
            return [255, 255, 255]  # 默认白色

class Lidar(Sensor):
    def __init__(self, range=100, accuracy=0.95, failure_rate=0.005, angular_resolution=1):
        super().__init__(range, accuracy, failure_rate)
        self.angular_resolution = angular_resolution

    def sense(self, agent, environment):
        self.check_failure()
        if not self.is_functioning:
            logger.warning(f"Lidar of agent {agent.id} is not functioning")
            return None

        num_rays = int(360 / self.angular_resolution)
        distances = np.full(num_rays, self.range)

        try:
            for i in range(num_rays):
                angle = i * self.angular_resolution * np.pi / 180
                direction = np.array([np.cos(angle), np.sin(angle)])
                ray_end = agent.position + direction * self.range

                for obj in environment.get_objects_in_range(agent.position, self.range):
                    if isinstance(obj, dict):
                        if 'center' in obj:
                            obj_position = np.array(obj['center'])
                        elif 'position' in obj:
                            obj_position = np.array(obj['position'])
                        else:
                            logger.warning(f"Object {obj} has no position information")
                            continue
                    else:
                        obj_position = obj.position
                    intersection = self.ray_intersection(agent.position, ray_end, obj_position)
                    if intersection is not None:
                        distance = np.linalg.norm(intersection - agent.position)
                        distances[i] = min(distances[i], distance)
        except Exception as e:
            logger.error(f"Error in Lidar sensing for agent {agent.id}: {str(e)}")
            return None

        return self.add_noise(distances)
    def ray_intersection(self, start, end, obj_position):
        # Simplified ray-object intersection
        # This assumes objects are points, which is not realistic but serves as a placeholder
        direction = end - start
        t = np.dot(obj_position - start, direction) / np.dot(direction, direction)
        if 0 <= t <= 1:
            intersection = start + t * direction
            if np.linalg.norm(intersection - obj_position) < 1:  # Assuming object has radius of 1
                return intersection
        return None

class Radar(Sensor):
    def __init__(self, range=200, accuracy=0.8, failure_rate=0.02, min_detection_size=1):
        super().__init__(range, accuracy, failure_rate)
        self.min_detection_size = min_detection_size

    def sense(self, agent, environment):
        self.check_failure()
        if not self.is_functioning:
            logger.warning(f"Radar of agent {agent.id} is not functioning")
            return None

        detected_objects = []

        for obj in environment.get_objects_in_range(agent.position, self.range):
            if isinstance(obj, dict):
                if 'center' in obj:
                    obj_position = np.array(obj['center'])
                elif 'position' in obj:
                    obj_position = np.array(obj['position'])
                else:
                    logger.warning(f"Object {obj} has no position information")
                    continue
                size = obj.get('radius', 0) if obj.get('type') == 'circle' else np.mean([np.linalg.norm(p) for p in obj.get('points', [])])
            else:
                obj_position = obj.position
                size = getattr(obj, 'size', 0)

            if size >= self.min_detection_size and np.random.random() < self.accuracy:
                relative_position = obj_position - agent.position
                relative_velocity = np.zeros(2)  # 假设静态对象，如果需要可以修改
                detected_objects.append({
                    'position': self.add_noise(relative_position),
                    'velocity': self.add_noise(relative_velocity),
                    'size': size
                })

        return detected_objects

class GPS(Sensor):
    def __init__(self, accuracy=0.95, failure_rate=0.001):
        super().__init__(range=float('inf'), accuracy=accuracy, failure_rate=failure_rate)

    def sense(self, agent, environment):
        self.check_failure()
        if not self.is_functioning:
            logger.warning(f"GPS of agent {agent.id} is not functioning")
            return None

        true_position = agent.position
        error = np.random.normal(0, 1 - self.accuracy, 2)
        return true_position + error