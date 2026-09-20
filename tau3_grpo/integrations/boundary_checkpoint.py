"""Publish a complete training boundary before removing the previous one."""
import json
import re
import shutil
from pathlib import Path

from tau3_grpo.tracking.rl_continuity import atomic_json
from tau3_grpo.training.rl.checkpoints import required_files, validate_checkpoint


def complete_boundary(root, step, world_size, keep=1):
    root = Path(root)
    current = root / f'global_step_{step}'
    files = [current / name for name in sorted(required_files(world_size))]
    if any(not path.is_file() or path.stat().st_size == 0 for path in files):
        raise ValueError('Incomplete model/optimizer/RNG/dataloader checkpoint; previous checkpoint retained')
    metadata = root / 'swanlab-run.json'
    if metadata.exists():
        atomic_json(current / metadata.name, json.loads(metadata.read_text()))
    atomic_json(current / 'checkpoint-complete.json', {'step': step, 'world_size': world_size,
                'files': {str(p.relative_to(current)): p.stat().st_size for p in files}})
    validate_checkpoint(root, step, world_size=world_size, require_latest=False)
    marker = root / 'latest_checkpointed_iteration.txt'
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(str(step))
    temporary.replace(marker)
    previous = []
    for path in root.glob('global_step_*'):
        match = re.fullmatch(r'global_step_(\d+)', path.name)
        if match and path.is_dir() and not path.is_symlink() and int(match[1]) <= step:
            previous.append((int(match[1]), path))
    if keep and keep > 0:
        for _, path in sorted(previous, reverse=True)[keep:]:
            shutil.rmtree(path)
