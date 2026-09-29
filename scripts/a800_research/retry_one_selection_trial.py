"""One explicitly authorized supplemental attempt; never edit original results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import urllib.request


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_failed_identity(run, errors, task_id, trial):
    if run['spec']['target'] != 'selection':
        raise ValueError('Only internal selection supplements are allowed')
    matches = [v for v in run['planned'] if v['task_id'] == task_id and v['trial'] == trial]
    if len(matches) != 1:
        raise ValueError('Expected exactly one original planned identity')
    identity = matches[0]
    failures = [v for v in errors if all(v.get(k) == identity[k] for k in ('task_id', 'trial', 'seed'))]
    if len(failures) != 1:
        raise ValueError('Requested identity is not exactly one recorded failed attempt')
    return identity, failures[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--trial', type=int, required=True)
    parser.add_argument('--user-base-url', default='http://127.0.0.1:8210/v1')
    args = parser.parse_args()
    source = args.source.resolve()
    parent = source.parent.parent
    run = json.loads((source / 'run.json').read_text())
    errors = [json.loads(line) for line in (source / 'errors.jsonl').read_text().splitlines() if line.strip()]
    identity, failure = select_failed_identity(run, errors, args.task_id, args.trial)
    if any(all(v.get(k) == identity[k] for k in identity)
           for v in (json.loads(line) for line in (source / 'trajectories.jsonl').read_text().splitlines() if line.strip())):
        raise ValueError('Identity already has an original scored result')
    code = Path(__file__).resolve().parents[2]
    actual_sources = {str(p.relative_to(code)): sha(p) for p in sorted((code / 'tau3_grpo/evaluation').rglob('*.py'))}
    if actual_sources != json.loads((parent / 'software.json').read_text())['evaluation_source_sha256']:
        raise ValueError('Evaluation implementation differs from original run')
    from tau3_grpo.data.manifest import read_manifest
    from tau3_grpo.evaluation.runtime import Endpoint, _selection_jobs, _run_one
    from tau3_grpo.evaluation.service_attestation import assert_service_matches_checkpoint
    from tau3_grpo.prompts import prompt_provenance
    for key, value in prompt_provenance().items():
        if run['provenance'].get(key) != value:
            raise ValueError('Prompt protocol differs from original run')
    manifest = parent / 'manifests' / f"areal_airline_selection_seed{run['provenance']['data_seed']}.jsonl"
    entries = [entry for entry in read_manifest(manifest) if entry.task_id == args.task_id]
    if len(entries) != 1:
        raise ValueError('Expected one frozen selection task')
    jobs = list(_selection_jobs(entries, run['spec']['trials'], run['spec']['seed']))
    job = next(v for v in jobs if all(v[k] == identity[k] for k in identity))
    attestation_path = parent / f'{source.name}-attestation.json'
    attestation_data = json.loads(attestation_path.read_text())
    checkpoint = run['provenance']['checkpoint']
    attestation = assert_service_matches_checkpoint(
        attestation_path=attestation_path, checkpoint_path=checkpoint,
        served_model_name=run['endpoints']['policy']['model'], base_url=attestation_data['base_url'])
    if attestation.checkpoint_hash != run['provenance']['checkpoint_hash']:
        raise ValueError('Checkpoint differs from original run')
    for endpoint, model, length in [(attestation.base_url, run['endpoints']['policy']['model'], 24576),
                                     (args.user_base_url, run['endpoints']['user']['model'], 16384)]:
        with urllib.request.urlopen(endpoint.rstrip('/') + '/models', timeout=10) as response:
            served = json.loads(response.read())['data']
        matches = [v for v in served if v['id'] == model]
        if len(matches) != 1 or matches[0].get('max_model_len') != length:
            raise ValueError('Live model/context limit differs from original service')
    # Exclusive output directory is the one-attempt guard, including partial runs.
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'running', 'attempt': 1, 'kind': 'supplemental_single_retry',
                'started_at': datetime.now(timezone.utc).isoformat(), 'identity': identity,
                'original_run': str(source), 'original_run_sha256': sha(source / 'run.json'),
                'original_failure': failure, 'checkpoint_hash': attestation.checkpoint_hash,
                'manifest_sha256': sha(manifest), 'evaluation_source_sha256': actual_sources,
                'original_spec': run['spec'], 'supplemental_concurrency': 1,
                'policy_context': 24576, 'user_context': 16384,
                'note': 'One new attempt, not an exact continuation; original error and aggregate validity are unchanged.'}
    def write(name, data):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    write('run.json', metadata)
    print(json.dumps({'status': 'running', **identity}), flush=True)
    policy = Endpoint(model=run['endpoints']['policy']['model'], base_url=attestation.base_url,
                      temperature=run['endpoints']['policy']['temperature'])
    user = Endpoint(model=run['endpoints']['user']['model'], base_url=args.user_base_url,
                    temperature=run['endpoints']['user']['temperature'])
    try:
        simulation = _run_one(task=job['task'], db_path=job['db_path'], policy=policy, user=user,
                              seed=identity['seed'], max_steps=run['spec']['max_steps'],
                              max_errors=run['spec']['max_errors'])
        if simulation.termination_reason.value == 'infrastructure_error':
            raise RuntimeError('Official simulator returned infrastructure_error')
        reward = float(simulation.reward_info.reward)
        if not math.isfinite(reward):
            raise ValueError('Nonfinite reward')
        row = {**identity, 'reward': reward, 'termination_reason': simulation.termination_reason.value,
               'simulation': simulation.model_dump(mode='json')}
        write('result.json', row)
        summary = {'status': 'scored', **identity, 'reward': reward, 'attempt': 1,
                   'termination_reason': row['termination_reason'], 'original_results_modified': False}
        exit_code = 0
    except Exception as exc:
        row = {**identity, 'error_type': type(exc).__name__, 'error': str(exc), 'attempt': 1}
        write('error.json', row)
        summary = {'status': 'failed', **row, 'original_results_modified': False}
        exit_code = 1
    summary['finished_at'] = datetime.now(timezone.utc).isoformat()
    write('summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
