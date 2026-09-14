"""Pin anchor protocol to a run directory before starting distributed workers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tau3_grpo.algorithms.anchors.evidence import validate_version


def pin_protocol(directory: Path, version: str):
    validate_version(version)
    directory=Path(directory)
    path=directory/'anchor_protocol.json'
    expected={'schema_version':1,'anchor_version':version}
    if path.exists():
        if json.loads(path.read_text())!=expected:
            raise ValueError('anchor protocol changed; use a new run directory for the new protocol')
        return path
    # Historical directories did not have a sidecar and used v1. Never silently
    # resume their checkpoint with a different credit-assignment protocol.
    historical=(directory/'experiment_manifest.json').exists() or any(directory.glob('global_step_*'))
    if historical and version!='v1':
        raise ValueError(f'historical run has v1 anchors; choose a new run directory for {version}')
    directory.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('x') as f:json.dump(expected,f,sort_keys=True)
    except FileExistsError:
        if json.loads(path.read_text())!=expected:raise ValueError('concurrent anchor protocol mismatch')
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results-dir',type=Path,required=True);p.add_argument('--version',required=True);a=p.parse_args()
    print(pin_protocol(a.results_dir,a.version))
