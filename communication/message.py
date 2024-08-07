import time
import uuid

class Message:
    def __init__(self, sender, receiver, content, priority=1):
        self.id = str(uuid.uuid4())  # Unique identifier for each message
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.priority = priority
        self.timestamp = time.time()
        self.size = len(str(content))  # Simple size estimation
        self.hops = 0  # Number of hops the message has taken
        self.ttl = 100  # Time to live, in number of hops
        self.created_at = time.time()
        self.delivered_at = None

    def __str__(self):
        return f"Message from {self.sender.id} to {self.receiver.id}: {self.content[:20]}..."

    def increment_hop(self):
        self.hops += 1
        return self.hops < self.ttl

    def is_expired(self):
        return self.hops >= self.ttl or (time.time() - self.created_at) > 3600  # 1 hour expiration

    def mark_delivered(self):
        self.delivered_at = time.time()

    def get_age(self):
        return time.time() - self.created_at

    def get_delivery_time(self):
        if self.delivered_at:
            return self.delivered_at - self.created_at
        return None

    def to_dict(self):
        return {
            'id': self.id,
            'sender': self.sender.id,
            'receiver': self.receiver.id,
            'content': self.content,
            'priority': self.priority,
            'timestamp': self.timestamp,
            'size': self.size,
            'hops': self.hops,
            'ttl': self.ttl,
            'created_at': self.created_at,
            'delivered_at': self.delivered_at
        }

    @classmethod
    def from_dict(cls, data, env):
        sender = env.get_agent_by_id(data['sender'])
        receiver = env.get_agent_by_id(data['receiver'])
        message = cls(sender, receiver, data['content'], data['priority'])
        message.id = data['id']
        message.timestamp = data['timestamp']
        message.size = data['size']
        message.hops = data['hops']
        message.ttl = data['ttl']
        message.created_at = data['created_at']
        message.delivered_at = data['delivered_at']
        return message