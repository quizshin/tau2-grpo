"""Run isolated full-RL acceptance, then the unchanged 40-update formal budget."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from string import Template
import time

import requests
from tau3_grpo.launch import load_config, prepare
from tau3_grpo.tracking.swanlab import load_tracking_env

R=Path(os.environ['TAU3_ROOT'])
C=R/'code'
W=R/'runs/rl-formal-full-20260912'
PROFILE=C/'configs/train/rl/qwen35_4b_full_a800_formal_20260912.yaml'
processes=[]

def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=45)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def launch(cmd,env,name):
    with (W/(name+'.log')).open('x') as log:
        p=subprocess.Popen(cmd,cwd=C,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    processes.append(p);(W/(name+'.pid')).write_text(str(p.pid));return p

def stage_config(stage):
    env=os.environ.copy()
    for key,value in load_config(PROFILE)['launch']['environment'].items():
        env[key]=Template(str(value)).substitute(env)
    extras=[]
    if stage=='acceptance':
        env.update(RESULTS_DIR=str(W/'acceptance'),GROUP_SIZE='4',GROUPS_PER_UPDATE='2',
                   PPO_MINI_GROUPS='2',TOTAL_UPDATES='2',
                   SWANLAB_EXPERIMENT_NAME='acceptance-E0-Qwen35-4B-full-newSFT-off-g2x4-u2-s42')
        for key,suffix in [('TOOL_CONFIG','tool_config.yaml'),('SWANLAB_LOG_DIR','swanlog'),
                           ('TAU3_GRPO_DEBUG_BATCH_DIR','update-batches')]:
            env[key]=str(Path(env['RESULTS_DIR'])/suffix)
        extras=['trainer.save_freq=1']
        env['VERL_QWEN35_WEIGHT_AUDIT_DIR']=str(W/'acceptance/weight-audits')
    env['GIT_CONFIG_COUNT']='1';env['GIT_CONFIG_KEY_0']='safe.directory';env['GIT_CONFIG_VALUE_0']=str(C)
    cmd,env,snapshot=prepare('rl',PROFILE,'e0',42,extras,env)
    assert env['MODEL_PATH']==str(R/'checkpoints/sft-merged/new-off')
    assert env['POLICY_GPUS']=='4' and env['ROLLOUT_TP']=='1'
    assert 'actor_rollout_ref.model.lora_rank=0' in cmd
    assert 'trainer.resume_mode=disable' in cmd
    dry_env=env.copy();dry_env['TAU3_DRY_RUN']='1'
    dry=subprocess.check_output(cmd,env=dry_env,cwd=C,text=True)
    assert 'tool_execution_mode=sequential' in dry
    assert 'enable_thinking=false' in dry
    (W/(stage+'.launch.json')).write_text(json.dumps(snapshot,indent=2))
    (W/(stage+'.command.txt')).write_text(dry)
    return cmd,env

def main(dry_run):
    load_tracking_env()
    stages={name:stage_config(name) for name in ['acceptance','formal']}
    if dry_run:
        print('Both configurations validated; no service or training started.');return
    with (W/'rl-controller.started').open('x') as f:f.write(str(time.time()))
    while not (W/'merge.exit').exists():time.sleep(10)
    assert (W/'merge.exit').read_text()=='0','Four-model merge failed'
    merged=json.loads((W/'merged-models.json').read_text())
    assert len(merged)==4 and all(x['all_tensors_finite'] for x in merged.values())
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert not active.strip(),'GPU compute processes already present; refusing to share devices'
    assert shutil.disk_usage(R).free>110*2**30,'Insufficient shared space for acceptance and formal checkpoints'
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    assert len(inventory.strip().splitlines())==5
    (W/'gpu-assignment.txt').write_text(inventory+'\npolicy=0,1,2,3; simulator=4\n')
    env=os.environ.copy()
    env.update(TAU3_USER_CUDA_DEVICES='4',TAU3_USER_MAX_NUM_SEQS='16',TAU3_USER_MAX_MODEL_LEN='16384',
               TAU3_USER_GPU_MEMORY_UTILIZATION='0.65',TAU3_USER_ENFORCE_EAGER='1',
               TAU3_USER_MODEL=str(R/'models/Qwen3.8-27B-AWQ-INT4'),
               TAU3_USER_SERVED_MODEL_NAME='Qwen/Qwen3.8-27B-AWQ-INT4',TAU3_USER_PORT='8100',
               OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    # Fail rather than accidentally connect to somebody else's server.
    import socket
    with socket.socket() as sock:sock.bind(('127.0.0.1',8100))
    simulator=launch(['bash',str(C/'scripts/serve/simulator_qwen38.sh')],env,'simulator')
    deadline=time.monotonic()+1200
    while True:
        if simulator.poll() is not None:raise RuntimeError('Simulator exited during startup')
        try:
            if requests.get('http://127.0.0.1:8100/health',timeout=3).status_code==200:break
        except requests.RequestException:pass
        if time.monotonic()>deadline:raise TimeoutError('Simulator startup timed out')
        time.sleep(3)
    for stage,(cmd,env) in stages.items():
        print('Starting',stage,flush=True)
        if stage=='formal':
            assert shutil.disk_usage(R).free>55*2**30,'Insufficient free space for formal full checkpoint'
        train=launch(cmd,env,stage)
        rc=train.wait();(W/(stage+'.exit')).write_text(str(rc))
        if rc:raise RuntimeError(f'{stage} exited {rc}; formal continuation stopped')
        if stage=='acceptance':
            checkpoint=W/'acceptance/global_step_2/actor'
            assert len(list(checkpoint.glob('model_world_size_4_rank_*.pt')))==4
            assert len(list(checkpoint.glob('optim_world_size_4_rank_*.pt')))==4
            log=(W/'acceptance.log').read_text()
            norms=[float(v) for v in re.findall(r'actor/grad_norm[^0-9+\-]*([+\-]?[0-9.]+(?:e[+\-]?\d+)?)',log)]
            assert norms and any(x>0 for x in norms),'No nonzero actor gradient recorded'
            from verl import DataProto
            batches=list((W/'acceptance/update-batches').glob('update_*.pkl'))
            assert len(batches)==2,'Missing acceptance update batches'
            advantage_checks=[]
            for path in batches:
                batch=DataProto.load_from_disk(str(path))
                advantages=batch.batch['advantages']
                import torch
                assert torch.isfinite(advantages).all(),'Nonfinite advantages'
                advantage_checks.append(bool((advantages!=0).any()))
            assert any(advantage_checks),'No nonzero reward advantage; engineering gate needs further investigation'
            (W/'acceptance-gate.json').write_text(json.dumps({'two_updates_completed':True,
                'model_and_optimizer_shards':4,'recorded_grad_norms':norms,
                'nonzero_advantages_by_update':advantage_checks,
                'note':'Engineering gate, not proof of benchmark improvement; formal restarts from SFT.'},indent=2))
        print('Completed',stage,flush=True)
    (W/'rl-controller.exit').write_text('0')

if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('RL controller interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    try:main(args.dry_run)
    except BaseException as exc:
        if not args.dry_run:(W/'rl-controller.exit').write_text('1')
        print(type(exc).__name__,str(exc),flush=True);raise
    finally:
        for p in reversed(processes):stop(p)
