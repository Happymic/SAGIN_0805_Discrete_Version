import numpy as np
from .protocols import ShortRangeProtocol, LongRangeProtocol

class CommunicationModel:
    def __init__(self, env):
        self.env = env
        self.short_range_protocol = ShortRangeProtocol()
        self.long_range_protocol = LongRangeProtocol()
        self.message_queue = []

    def send_message(self, sender, receiver, content, priority=1):
        message = Message(sender, receiver, content, priority)
        if self.can_communicate(sender, receiver):
            self.deliver_message(message)
            return True
        else:
            self.message_queue.append(message)
            return False

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
        if np.random.random() > self.get_packet_loss_rate(message.sender, message.receiver):
            delay = self.calculate_delay(message)
            self.env.schedule_event(self.env.time + delay, 'message_received', message)

    def get_packet_loss_rate(self, sender, receiver):
        distance = np.linalg.norm(sender.position - receiver.position)
        if distance <= self.short_range_protocol.range:
            return self.short_range_protocol.packet_loss_rate
        else:
            return self.long_range_protocol.packet_loss_rate

    def calculate_delay(self, message):
        distance = np.linalg.norm(message.sender.position - message.receiver.position)
        if distance <= self.short_range_protocol.range:
            return self.short_range_protocol.calculate_delay(message.size)
        else:
            return self.long_range_protocol.calculate_delay(message.size)

    def update(self):
        # Try to send queued messages
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