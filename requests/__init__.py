class Response:
    def __init__(self, json_data=None, status_code=200):
        self._json = json_data or {}
        self.status_code = status_code
    def json(self):
        return self._json
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")

def post(url, files=None, data=None, json=None, params=None, timeout=None):
    raise NotImplementedError

def get(url, params=None, timeout=None):
    raise NotImplementedError
