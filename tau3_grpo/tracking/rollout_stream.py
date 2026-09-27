"""Per-completion artifacts; deliberately not a batch-resume/checkpoint format."""
import asyncio
import json
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path


def _json_default(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Unsupported rollout artifact value: {type(value).__name__}')


def _write(root, trajectory, output):
    directory = Path(root) / ('validation' if trajectory['validate'] else 'training') / str(trajectory['step'])
    directory.mkdir(parents=True, exist_ok=True)
    # UUID distinguishes worker-local indices, repeated evaluations and resumed sessions.
    artifact_id = uuid.uuid4().hex
    payload = dict(schema='tau3_completed_rollout_v1', artifact_id=artifact_id,
                   completed_at=time.time(), host=socket.gethostname(), pid=os.getpid(),
                   trajectory=trajectory, stage='agent_postprocess_complete',
                   batch_complete=False, resumable_checkpoint=False,
                   output=output.model_dump())
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=_json_default)
    path = directory / f'{artifact_id}.json'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                         prefix='.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


async def save_completed_rollout(trajectory, output):
    root = os.environ.get('TAU3_ROLLOUT_STREAM_DIR')
    if not root:
        return None
    # Await persistence before returning this trajectory to gather. Disk errors propagate.
    return await asyncio.to_thread(_write, root, trajectory, output)
