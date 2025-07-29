import json
from hashmancer.utils import event_logger

class FakeRedis:
    def __init__(self):
        self.data = []
    def rpush(self, name, value):
        self.data.append(value)
    def ltrim(self, name, start, end):
        pass

def test_log_event_stores_provided_traceback(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(event_logger, "r", fake)
    tb = "Traceback: something bad"
    event_logger.log_event("w", "E1", "msg", level="error", details=tb)
    assert fake.data, "no data pushed"
    stored = json.loads(fake.data[0])
    assert stored["traceback"] == tb

