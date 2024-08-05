import numpy as np

class CommunicationProtocol:
    def __init__(self, range, bandwidth, base_delay, packet_loss_rate):
        self.range = range
        self.bandwidth = bandwidth  # in bits per second
        self.base_delay = base_delay
        self.packet_loss_rate = packet_loss_rate

    def calculate_delay(self, message_size):
        transmission_delay = message_size / self.bandwidth
        return self.base_delay + transmission_delay

class ShortRangeProtocol(CommunicationProtocol):
    def __init__(self):
        super().__init__(range=100, bandwidth=1e6, base_delay=0.001, packet_loss_rate=0.01)

class LongRangeProtocol(CommunicationProtocol):
    def __init__(self):
        super().__init__(range=1000, bandwidth=1e5, base_delay=0.01, packet_loss_rate=0.05)

    def calculate_delay(self, message_size):
        base_delay = super().calculate_delay(message_size)
        # Add random delay to simulate atmospheric interference
        atmospheric_delay = np.random.exponential(0.05)
        return base_delay + atmospheric_delay