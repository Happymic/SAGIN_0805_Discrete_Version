# 如果这个文件不存在，创建它并添加以下内容：
from collections import defaultdict

class Event:
    def __init__(self, time, event_type, data):
        self.time = time
        self.event_type = event_type
        self.data = data

class EventScheduler:
    def __init__(self):
        self.events = defaultdict(list)

    def schedule_event(self, time, event_type, data):
        self.events[time].append(Event(time, event_type, data))

    def get_current_events(self, current_time):
        current_events = self.events.pop(current_time, [])
        return current_events

    def clear(self):
        self.events.clear()