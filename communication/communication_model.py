import numpy as np
from .protocols import ShortRangeProtocol, LongRangeProtocol
from .message import Message
from .security import encrypt_message, decrypt_message

class CommunicationModel:
    def __init__(self, env):
        self.env = env
        self.short_range_protocol = ShortRangeProtocol()
        self.long_range_protocol = LongRangeProtocol()
        self.message_queue = []
        self.interference_level = 0.0
        self.relay_counts = {}  # To keep track of relay counts for each agent

    def send_message(self, sender, receiver, content, priority=1):
        if not self.can_communicate(sender, receiver):
            return self.store_message_for_dtn(sender, receiver, content, priority)

        encrypted_content = encrypt_message(content)
        message = Message(sender, receiver, encrypted_content, priority)

        if np.random.random() < self.interference_level:
            return False  # Message lost due to interference

        self.deliver_message(message)
        return True

    def can_communicate(self, sender, receiver):
        distance = np.linalg.norm(sender.position - receiver.position)
        if distance <= self.short_range_protocol.range:
            return not self.is_obstructed(sender, receiver)
        elif distance <= self.long_range_protocol.range:
            return True
        return False

    def is_obstructed(self, sender, receiver):
        # Check if there's any obstacle between sender and receiver
        for obstacle in self.env.world.obstacles:
            if self.line_intersects_obstacle(sender.position, receiver.position, obstacle):
                return True
        return False

    def line_intersects_obstacle(self, start, end, obstacle):
        if obstacle['type'] == 'circle':
            # Implement circle-line intersection check
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            direction = end - start
            f = start - center
            a = np.dot(direction, direction)
            b = 2 * np.dot(f, direction)
            c = np.dot(f, f) - radius * radius
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                return False
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2 * a)
            t2 = (-b + discriminant) / (2 * a)
            if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                return True
            return False
        elif obstacle['type'] == 'polygon':
            # Implement polygon-line intersection check
            # This is a simplified check and might not work for all cases
            points = np.array(obstacle['points'])
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                if self.line_segments_intersect(start, end, p1, p2):
                    return True
            return False
        return False

    def line_segments_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def deliver_message(self, message):
        delay = self.calculate_delay(message)
        self.env.schedule_event(self.env.time + delay, 'message_received', message)

    def store_message_for_dtn(self, sender, receiver, content, priority):
        message = Message(sender, receiver, content, priority)
        self.message_queue.append(message)
        return False

    def calculate_delay(self, message):
        distance = np.linalg.norm(message.sender.position - message.receiver.position)
        if distance <= self.short_range_protocol.range:
            return self.short_range_protocol.calculate_delay(message.size)
        else:
            return self.long_range_protocol.calculate_delay(message.size)

    def update(self):
        self.update_interference()
        self.attempt_dtn_delivery()

    def update_interference(self):
        self.interference_level = np.random.uniform(0, 0.2)  # Random interference level

    def attempt_dtn_delivery(self):
        remaining_messages = []
        for message in self.message_queue:
            if self.can_communicate(message.sender, message.receiver):
                self.deliver_message(message)
                self.increment_relay_count(message.sender)
            else:
                remaining_messages.append(message)
        self.message_queue = remaining_messages

    def broadcast(self, sender, content, range):
        receivers = [agent for agent in self.env.agents if agent != sender and
                     np.linalg.norm(agent.position - sender.position) <= range]
        for receiver in receivers:
            self.send_message(sender, receiver, content)

    def get_undelivered_messages(self):
        return self.message_queue

    def get_status(self):
        return {
            "num_messages": len(self.message_queue),
            "interference_level": self.interference_level
        }

    def get_messages_for(self, agent):
        return [msg for msg in self.message_queue if msg.receiver == agent]

    def remove_message(self, message):
        self.message_queue.remove(message)

    def increment_relay_count(self, agent):
        if agent in self.relay_counts:
            self.relay_counts[agent] += 1
        else:
            self.relay_counts[agent] = 1

    def get_relay_count(self, agent):
        return self.relay_counts.get(agent, 0)

    def reset(self):
        self.message_queue.clear()
        self.interference_level = 0.0
        self.relay_counts.clear()