"""Run one fresh online 8x8 RL update with the validated FLA IEEE path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
from string import Template

from tau3_grpo.launch import load_config,prepare
from tau3_grpo.paths import CODE_ROOT
import run_curriculum_smoke as controller

R=Path(os.environ['TAU3_ROOT'])
W=Path(os.environ.get('TAU3_FLA_ONLINE_DIR',str(R / 'code/results/runs/fla-online-20260912')))


def resolved(stage='smoke'):
    assert stage=='smoke'
    profile=CODE_ROOT/'configs/train/rl/qwen35_4b_full_a800_fla_online_smoke_20260912.yaml'
    env=os.environ.copy();env['CODE_ROOT']=str(CODE_ROOT);env.pop('TRAIN_PARQUET',None)
    for key,value in load_config(profile)['launch']['environment'].items():
        env[key]=Template(str(value)).substitute(env)
    result=W/'online-smoke'
    env.update(RESULTS_DIR=str(result),TOTAL_UPDATES='1',TOOL_CONFIG=str(result/'tool_config.yaml'),
        INTERACTION_CONFIG=str(result/'interaction_config.yaml'),SWANLAB_LOG_DIR=str(result/'swanlog'),
        TAU3_GRPO_DEBUG_BATCH_DIR=str(result/'update-batches'),VERL_QWEN35_WEIGHT_AUDIT_DIR=str(result/'weight-audits'))
    command,env,snapshot=prepare('rl',profile,'e0',42,[],env)
    assert 'fa2-overlay' not in env.get('PYTHONPATH','')
    assert env['VERL_QWEN35_FLA_IEEE']=='1' and env['VERL_QWEN35_TRIM_PADDING']=='experimental_both'
    manifest=Path(env['TRAIN_MANIFEST_DIR'])/'areal_airline_train_seed42.jsonl'
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest=='79e317a824e8b55f7ce338c9f30a5811d83f2dce17dd896397eaffb003f5efbb'
    (W/'smoke.launch.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    return command,env,snapshot


def main(dry_run):
    W.mkdir(parents=True,exist_ok=True)
    old=R / 'code/results/runs/curriculum-speed-20260912/data-audit.json'
    audit=json.loads(old.read_text())
    assert audit['candidate_manifest_sha256']=='79e317a824e8b55f7ce338c9f30a5811d83f2dce17dd896397eaffb003f5efbb'
    (W/'data-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    controller.W=W;controller.resolved=resolved
    try:
        controller.main(dry_run,False)
        if not dry_run:
            log=(W/'online.log').read_text()
            assert '"tau3_fla_ieee": "installed"' in log
            assert '"phase": "backprop_enabled"' in log
            result=json.loads((W/'online-result.json').read_text())
            result.update(fla_ieee=True,fa2=False,fp32_compute=True,
                actor_param_optimizer_offload=False,reference_param_offload=True)
            (W/'online-result.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        for child in reversed(controller.owned):controller.stop(child)


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('FLA online controller interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true')
    main(parser.parse_args().dry_run)
