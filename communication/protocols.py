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

    def is_message_lost(self):
        return np.random.random() < self.packet_loss_rate

class ShortRangeProtocol(CommunicationProtocol):
    def __init__(self):
        super().__init__(range=100, bandwidth=1e6, base_delay=0.001, packet_loss_rate=0.01)

    def calculate_delay(self, message_size):
        base_delay = super().calculate_delay(message_size)
        # Add some small random variation to the delay
        return base_delay + np.random.exponential(0.001)

class LongRangeProtocol(CommunicationProtocol):
    def __init__(self):
        super().__init__(range=1000, bandwidth=1e5, base_delay=0.01, packet_loss_rate=0.05)

    def calculate_delay(self, message_size):
        base_delay = super().calculate_delay(message_size)
        # Add random delay to simulate atmospheric interference
        atmospheric_delay = np.random.exponential(0.05)
        return base_delay + atmospheric_delay

class SatelliteProtocol(CommunicationProtocol):
    def __init__(self):
        super().__init__(range=10000, bandwidth=1e4, base_delay=0.1, packet_loss_rate=0.02)

    def calculate_delay(self, message_size):
        base_delay = super().calculate_delay(message_size)
        # Add significant delay to simulate the long distance to satellites
        satellite_delay = np.random.uniform(0.1, 0.5)
        return base_delay + satellite_delay

def get_protocol(agent_type):
    if agent_type in ['ground', 'fixed']:
        return ShortRangeProtocol()
    elif agent_type == 'uav':
        return LongRangeProtocol()
    elif agent_type == 'satellite':
        return SatelliteProtocol()
    else:
        return ShortRangeProtocol()  # Default to short range protocol