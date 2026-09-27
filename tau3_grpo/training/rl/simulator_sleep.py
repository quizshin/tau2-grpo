"""Phase-controlled memory sharing with an explicitly owned local simulator."""
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


class SimulatorSleep:
    def __init__(self, base_url, event_path, *, session=None):
        parsed = urlparse(base_url)
        if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
                or parsed.path != '/v1' or parsed.query or parsed.fragment):
            raise ValueError('Simulator memory sharing requires the owned local /v1 endpoint')
        self.url = base_url.removesuffix('/v1')
        self.event_path = Path(event_path)
        self.session = session or requests.Session()

    def set_sleeping(self, sleeping):
        response = self.session.get(self.url + '/is_sleeping', timeout=15)
        response.raise_for_status()
        before = response.json()['is_sleeping']
        if type(before) is not bool:
            raise ValueError('Invalid simulator sleep status')
        started = time.monotonic()
        if before != sleeping:
            endpoint = '/sleep' if sleeping else '/wake_up'
            options = {'params': {'level': 1, 'mode': 'wait'}} if sleeping else {}
            response = self.session.post(self.url + endpoint, timeout=300, **options)
            response.raise_for_status()
        response = self.session.get(self.url + '/is_sleeping', timeout=15)
        response.raise_for_status()
        if response.json()['is_sleeping'] is not sleeping:
            raise RuntimeError('Simulator did not reach the requested memory state')
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open('a') as handle:
            handle.write(json.dumps({'time': time.time(), 'sleeping': sleeping,
                'changed': before != sleeping, 'seconds': time.monotonic() - started}) + '\n')
