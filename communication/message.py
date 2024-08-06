import time

class Message:
    def __init__(self, sender, receiver, content, priority=1):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.priority = priority
        self.timestamp = time.time()
        self.size = len(str(content))  # Simple size estimation

    def __str__(self):
        return f"Message from {self.sender.id} to {self.receiver.id}: {self.content[:20]}..."