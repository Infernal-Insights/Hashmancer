class RedisError(Exception):
    pass

class Redis:
    def __init__(self, host='localhost', port=6379, decode_responses=False):
        self.store = {}

    def hset(self, key, mapping=None, **kwargs):
        if mapping:
            self.store.setdefault(key, {}).update(mapping)
        else:
            field = kwargs.get('field')
            value = kwargs.get('value')
            self.store.setdefault(key, {})[field] = value

    def hget(self, key, field):
        return self.store.get(key, {}).get(field)

    def sadd(self, key, *values):
        self.store.setdefault(key, set()).update(values)

    def scard(self, key):
        val = self.store.get(key)
        return len(val) if isinstance(val, set) else 0

    def smembers(self, key):
        val = self.store.get(key)
        return val if isinstance(val, set) else set()

    def set(self, key, value):
        self.store[key] = value

class exceptions:
    RedisError = RedisError
