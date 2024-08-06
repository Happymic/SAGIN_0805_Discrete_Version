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
            pass
        elif obstacle['type'] == 'polygon':
            # Implement polygon-line intersection check
            pass
        return False

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

    # 添加 get_status 方法：
    def get_status(self):
        return {
            "num_messages": len(self.message_queue),
            "interference_level": self.interference_level
        }
    def get_messages_for(self, agent):
        return [msg for msg in self.message_queue if msg.receiver == agent]

    def remove_message(self, message):
        self.message_queue.remove(message)