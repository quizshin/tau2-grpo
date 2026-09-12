"""One 64-trajectory online E0 engineering step, then stop all owned services.

Never continues into the 640-trajectory pilot or the 3,200-trajectory extension.
No model checkpoint is written. Timings and rollout/update batches are retained.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import time

import requests
from prepare_curriculum_speed import CODE_ROOT, R, W, resolved

owned=[]


def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL)
            p.wait()


def launch(command,env,name):
    with (W/f'{name}.log').open('x') as log:
        p=subprocess.Popen(command,env=env,cwd=CODE_ROOT,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    owned.append(p)
    (W/f'{name}.pid').write_text(str(p.pid))
    return p


def active_pids(devices):
    output=subprocess.check_output(['nvidia-smi','-i',devices,'--query-compute-apps=pid',
        '--format=csv,noheader'],text=True)
    return set(int(x) for x in output.splitlines() if x.strip())


def check_owned_benchmarks(pids):
    from pathlib import Path
    for pid in pids:
        try:
            command=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().replace('\0',' ')
        except FileNotFoundError:
            continue
        assert 'env_info/a800_20260912/performance_probe.py' in command, 'Unrelated policy GPU process present'


def main(dry_run,wait_for_benchmark):
    command,env,snapshot=resolved('smoke')
    env['TRAINER_LOGGERS']='[console]'
    env['GIT_CONFIG_COUNT']='1'
    env['GIT_CONFIG_KEY_0']='safe.directory'
    env['GIT_CONFIG_VALUE_0']=str(CODE_ROOT)
    # A single one-off test must not reactivate a previous job on restart.
    assert int(env['TOTAL_UPDATES'])==1
    assert 'trainer.save_freq=-1' in command
    dry=env.copy();dry['TAU3_DRY_RUN']='1'
    expanded=subprocess.check_output(command,env=dry,cwd=CODE_ROOT,text=True)
    (W/'smoke.command.txt').write_text(expanded)
    assert 'tool_execution_mode=sequential' in expanded
    assert 'enable_thinking=false' in expanded
    if dry_run:
        output=subprocess.check_output(command+['--cfg','job'],env=env,cwd=CODE_ROOT,text=True)
        (W/'smoke.hydra.yaml').write_text(output)
        print('Resolved one-step 8x8 smoke; no GPU service or RL started.')
        return
    audit=json.loads((W/'data-audit.json').read_text())
    assert audit['all_40_best_sft_links_present_in_actual_new_off_input']
    assert not active_pids('4'),'Simulator GPU occupied'
    active=active_pids('0,1,2,3')
    if active:
        assert wait_for_benchmark,'Policy GPU occupied'
        check_owned_benchmarks(active)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',8100))
    (W/'online.started').open('x').close()
    simenv=env.copy()
    simenv.update(TAU3_USER_CUDA_DEVICES='4',TAU3_USER_MAX_NUM_SEQS='16',
        TAU3_USER_MAX_MODEL_LEN='16384',TAU3_USER_GPU_MEMORY_UTILIZATION='.65',
        TAU3_USER_ENFORCE_EAGER='1',TAU3_USER_PORT='8100')
    simulator=launch(['bash',str(CODE_ROOT/'scripts/serve/simulator_qwen38.sh')],
        simenv,'simulator')
    deadline=time.monotonic()+1200
    while True:
        if simulator.poll() is not None:
            raise RuntimeError('Simulator failed to start')
        try:
            if requests.get('http://127.0.0.1:8100/health',timeout=3).status_code==200:
                break
        except requests.RequestException:
            pass
        if time.monotonic()>deadline:
            raise TimeoutError('Simulator startup timeout')
        time.sleep(3)
    deadline=time.monotonic()+1200
    while active_pids('0,1,2,3'):
        assert wait_for_benchmark
        check_owned_benchmarks(active_pids('0,1,2,3'))
        if time.monotonic()>deadline:
            raise TimeoutError('Own benchmark has not released policy GPUs')
        time.sleep(3)
    start=time.monotonic()
    train=launch(command,env,'online')
    rc=train.wait()
    (W/'online.exit').write_text(str(rc))
    if rc:
        raise RuntimeError(f'Online smoke exited {rc}')
    log=(W/'online.log').read_text()
    assert not list((W/'online-smoke').glob('global_step_*/actor'))
    from verl import DataProto
    import torch
    packets=list((W/'online-smoke/update-batches').glob('update_*.pkl'))
    assert len(packets)==1
    batch=DataProto.load_from_disk(str(packets[0])).batch
    assert len(batch)==64
    assert torch.isfinite(batch['advantages']).all()
    assert batch['advantages'].count_nonzero()>0,'No informative advantage in this diagnostic batch'
    metrics={'kind':'one_online_step_64_trajectories_not_formal_pilot',
        'train_process_wall_seconds':time.monotonic()-start,'trajectories':len(batch),
        'nonzero_advantage_tokens':int(batch['advantages'].count_nonzero()),
        'checkpoint_written':False,'automatic_continuation':False}
    (W/'online-result.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Controller interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser()
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--wait-for-benchmark',action='store_true')
    args=p.parse_args()
    try:
        main(args.dry_run,args.wait_for_benchmark)
    finally:
        for child in reversed(owned):
            stop(child)
