from collections import defaultdict
import heapq

class Event:
    def __init__(self, time, event_type, data):
        self.time = time
        self.event_type = event_type
        self.data = data

    def __lt__(self, other):
        return self.time < other.time

class EventScheduler:
    def __init__(self):
        self.events = []
        self.current_time = 0

    def schedule_event(self, time, event_type, data):
        heapq.heappush(self.events, Event(time, event_type, data))

    def get_current_events(self):
        current_events = []
        while self.events and self.events[0].time <= self.current_time:
            current_events.append(heapq.heappop(self.events))
        return current_events

    def update(self, time_step):
        self.current_time += time_step

    def clear(self):
        self.events.clear()
        self.current_time = 0

    def get_next_event_time(self):
        if self.events:
            return self.events[0].time
        return float('inf')

    def has_events(self):
        return len(self.events) > 0

    def __len__(self):
        return len(self.events)

    def __str__(self):
        return f"EventScheduler(current_time={self.current_time}, pending_events={len(self.events)})"