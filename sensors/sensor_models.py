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
        self.is_functioning = np.random.random() >= self.failure_rate

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

        try:
            objects = environment.get_objects_in_range(agent.position, self.range, agent.get_agent_type())
            logger.debug(f"Camera sensing {len(objects)} objects for agent {agent.id}")
            image = np.zeros((*self.resolution, 3), dtype=np.uint8)  # RGB image

            for obj in objects:
                if np.random.random() < self.accuracy:
                    color = self.get_object_color(obj)
                    obj_position = self.get_object_position(obj)
                    if obj_position is not None:
                        # 确保 obj_position 是 3D 的
                        if len(obj_position) == 2:
                            obj_position = np.append(obj_position, 0)
                        position = self.world_to_image(obj_position - agent.position)
                        self.draw_object(image, position, color)

            return self.add_noise(image.astype(float) / 255.0)  # Normalize to [0, 1]
        except Exception as e:
            logger.error(f"Error in Camera sensing for agent {agent.id}: {str(e)}")
            return None
    def world_to_image(self, position):
        x = int((position[0] + self.range) / (2 * self.range) * self.resolution[0])
        y = int((position[1] + self.range) / (2 * self.range) * self.resolution[1])
        return np.array([x, y])

    def draw_object(self, image, position, color):
        x, y = position.astype(int)
        x_min, x_max = max(0, x - 2), min(self.resolution[0], x + 3)
        y_min, y_max = max(0, y - 2), min(self.resolution[1], y + 3)
        image[y_min:y_max, x_min:x_max] = color

    def get_object_color(self, obj):
        colors = {
            'obstacle': np.array([128, 128, 128]),
            'agent': np.array([0, 255, 0]),
            'task': np.array([255, 0, 0])
        }
        if isinstance(obj, dict):
            return colors.get(obj.get('type', 'unknown'), np.array([255, 255, 255]))
        elif hasattr(obj, 'type'):
            return colors.get(obj.type, np.array([255, 255, 255]))
        else:
            return np.array([255, 255, 255])  # Default white

    def get_object_position(self, obj):
        if isinstance(obj, dict):
            if 'center' in obj:
                return np.array(obj['center'], dtype=float)
            elif 'position' in obj:
                return np.array(obj['position'], dtype=float)
            elif 'points' in obj:
                return np.mean(obj['points'], axis=0)
        elif hasattr(obj, 'position'):
            return np.array(obj.position, dtype=float)
        return np.zeros(3)  # Default 3D position

class Lidar(Sensor):
    def __init__(self, range=100, accuracy=0.95, failure_rate=0.005, angular_resolution=1):
        super().__init__(range, accuracy, failure_rate)
        self.angular_resolution = angular_resolution

    def sense(self, agent, environment):
        self.check_failure()
        if not self.is_functioning:
            logger.warning(f"Lidar of agent {agent.id} is not functioning")
            return None

        try:
            num_rays = int(360 / self.angular_resolution)
            distances = np.full(num_rays, self.range)

            objects = environment.get_objects_in_range(agent.position, self.range, agent.get_agent_type())

            for i in range(num_rays):
                angle = i * self.angular_resolution * np.pi / 180
                direction = np.array([np.cos(angle), np.sin(angle), 0])
                ray_end = agent.position + direction * self.range

                for obj in objects:
                    obj_position = self.get_object_position(obj)
                    if obj_position is not None:
                        intersection = self.ray_intersection(agent.position, ray_end, obj_position)
                        if intersection is not None:
                            distance = np.linalg.norm(intersection - agent.position)
                            distances[i] = min(distances[i], distance)

            return self.add_noise(distances)
        except Exception as e:
            logger.error(f"Error in Lidar sensing for agent {agent.id}: {str(e)}")
            return None

    def ray_intersection(self, start, end, obj_position):
        direction = end - start
        t = np.dot(obj_position - start, direction) / np.dot(direction, direction)
        if 0 <= t <= 1:
            intersection = start + t * direction
            if np.linalg.norm(intersection - obj_position) < 1:  # Assuming object has radius of 1
                return intersection
        return None

    def get_object_position(self, obj):
        if isinstance(obj, dict):
            if 'center' in obj:
                return np.array(obj['center'], dtype=float)
            elif 'position' in obj:
                return np.array(obj['position'], dtype=float)
            elif 'points' in obj:
                return np.mean(obj['points'], axis=0)
        elif hasattr(obj, 'position'):
            return np.array(obj.position, dtype=float)
        return np.zeros(3)  # Default 3D position

class Radar(Sensor):
    def __init__(self, range=200, accuracy=0.8, failure_rate=0.02, min_detection_size=1):
        super().__init__(range, accuracy, failure_rate)
        self.min_detection_size = min_detection_size

    def sense(self, agent, environment):
        self.check_failure()
        if not self.is_functioning:
            logger.warning(f"Radar of agent {agent.id} is not functioning")
            return None

        try:
            detected_objects = []
            objects = environment.get_objects_in_range(agent.position, self.range, agent.get_agent_type())

            for obj in objects:
                obj_position = self.get_object_position(obj)
                if obj_position is None:
                    continue

                size = self.get_object_size(obj)

                if size >= self.min_detection_size and np.random.random() < self.accuracy:
                    relative_position = obj_position - agent.position
                    relative_velocity = np.zeros(3)  # Assuming static objects, modify if needed
                    detected_objects.append({
                        'position': self.add_noise(relative_position),
                        'velocity': self.add_noise(relative_velocity),
                        'size': size
                    })

            return detected_objects
        except Exception as e:
            logger.error(f"Error in Radar sensing for agent {agent.id}: {str(e)}")
            return None

    def get_object_position(self, obj):
        if isinstance(obj, dict):
            if 'center' in obj:
                return np.array(obj['center'], dtype=float)
            elif 'position' in obj:
                return np.array(obj['position'], dtype=float)
            elif 'points' in obj:
                return np.mean(obj['points'], axis=0)
        elif hasattr(obj, 'position'):
            return np.array(obj.position, dtype=float)
        return np.zeros(3)  # Default 3D position

    def get_object_size(self, obj):
        if isinstance(obj, dict):
            if 'radius' in obj:
                return obj['radius']
            elif 'points' in obj:
                return np.mean([np.linalg.norm(p) for p in obj['points']])
            else:
                return self.min_detection_size
        elif hasattr(obj, 'size'):
            return obj.size
        else:
            return self.min_detection_size

class GPS(Sensor):
    def __init__(self, accuracy=0.95, failure_rate=0.001):
        super().__init__(range=float('inf'), accuracy=accuracy, failure_rate=failure_rate)

    def sense(self, agent, environment):
        self.check_failure()
        if not self.is_functioning:
            logger.warning(f"GPS of agent {agent.id} is not functioning")
            return None

        true_position = np.array(agent.position, dtype=float)
        error = np.random.normal(0, 1 - self.accuracy, 3)
        return true_position + error