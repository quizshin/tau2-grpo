"""Isolated, budget-matched evaluation of the two multicall SFT adapters."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

import requests
from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.evaluation.service_attestation import write_service_attestation
from tau3_grpo.prompts import prompt_provenance

R = Path('/root/autodl-fs/tau3-core')
W = R / 'code/results/legacy/experiments/multicall-aa267bb/selection'
C = W/'code'
M = Path('/dev/shm/tau3-multicall-aa267bb-selection')
PY = str(R/'environment/venvs/qwen35/bin/python')
SIM = str(R/'environment/venvs/qwen38-sim/bin/python')
GPUS = {'policy':'GPU-63d13965-cbe4-900f-b057-25d52f285032',
        'simulator':'GPU-ff283ad1-81a9-85e0-014f-7a28f5c8aed5'}
GPU_INDICES = {'policy':'2','simulator':'3'}
os.environ.update(OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',VLLM_NO_USAGE_STATS='1',
                  VLLM_CACHE_ROOT=str(W/'cache'),PYTHONUNBUFFERED='1')
processes=[]

def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def launch(cmd,name,device=None):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=GPU_INDICES[device] if device else ''
    (W/(name+'.command.json')).write_text(json.dumps(cmd,indent=2))
    with (W/(name+'.log')).open('x') as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,cwd=C)
    processes.append(p)
    (W/(name+'.pid')).write_text(str(p.pid))
    return p

def ready(p,port):
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        if p.poll() is not None:raise RuntimeError(f'service {port} exited {p.returncode}')
        try:
            if requests.get(f'http://127.0.0.1:{port}/health',timeout=3).status_code==200:return
        except requests.RequestException:pass
        time.sleep(3)
    raise TimeoutError(f'service {port} startup timed out')

def serve(python,model,name,port,memory,context,thinking,sim=False):
    cmd=[python,'-m','vllm.entrypoints.cli.main','serve',str(model),
         '--served-model-name',name,'--host','127.0.0.1','--port',str(port),
         '--tensor-parallel-size','1','--dtype','bfloat16','--max-model-len',str(context),
         '--max-num-seqs','2','--gpu-memory-utilization',str(memory),'--enforce-eager',
         '--language-model-only','--default-chat-template-kwargs',json.dumps({'enable_thinking':thinking}),
         '--override-generation-config',json.dumps({'max_new_tokens':2048})]
    if sim:cmd+=['--quantization','compressed-tensors']
    else:cmd+=['--enable-auto-tool-choice','--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3']
    return cmd

def interrupted(signum,frame):raise KeyboardInterrupt('Controller interrupted')
signal.signal(signal.SIGTERM,interrupted)

def main():
    for port in [8200,8300]:
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    assert all(uuid not in active for uuid in GPUS.values()),'Requested GPU already occupied'
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping=dict(row.split(', ',1) for row in inventory.strip().splitlines())
    assert all(mapping[GPU_INDICES[role]]==uuid for role,uuid in GPUS.items()),'GPU index mapping changed'
    assert shutil.disk_usage('/dev/shm').free>25*2**30,'Insufficient temporary RAM disk space'
    assert shutil.disk_usage(W).free>2*2**30,'Insufficient results disk space'
    reference=json.loads((R/'selection-dual-gpu/results-off/run.json').read_text())
    manifest=R/'code/data/manifests/areal_airline_selection_seed42.jsonl'
    entries=read_manifest(manifest)
    assert len(entries)==60
    planned=[dict(task_id=e.task_id,trial=t,seed=42+t) for e in entries for t in range(4)]
    assert planned==reference['planned'],'Task/seed order differs from baseline'
    for k,v in prompt_provenance().items():assert reference['provenance'][k]==v
    expected={'include_pass_hat':True,'ks':[1,2,4],'max_concurrency':2,'max_errors':10,
              'max_steps':30,'seed':42,'target':'selection','trials':4}
    assert reference['spec']==expected
    (W/'reference-run.json').write_text(json.dumps(reference,indent=2))
    plan={'policy_gpu':2,'simulator_gpu':3,'gpu_uuids':GPUS,'reserved_other_thread':[0,1],
          'methods':['off','on'],'tasks':60,'trials':4,'concurrency':2,'policy_temperature':.4,
          'user_temperature':0,'max_new_tokens_per_turn':2048,'policy_context':24576,
          'user_context':16384,'max_steps':30,'seed':42,'thinking_on_changes_inference_mode':True,
          'reference_thread':'01a08c2c-2828-7221-a696-8ca571971d37',
          'reference_results':str(R/'selection-dual-gpu'),'temporary_merged_models':str(M),
          'temporary_models_lost_on_reboot':True,'adapter_source':str(R / 'code/results/legacy/experiments/multicall-aa267bb/sft'),
          'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),**prompt_provenance()}
    (W/'plan.json').write_text(json.dumps(plan,indent=2))
    with (W/'controller.started').open('x') as f:f.write(str(time.time()))
    M.mkdir(exist_ok=True)
    for mode in ['off','on']:
        checkpoint=M/('merged-'+mode)
        if checkpoint.exists():
            assert (W/'startup-failure-1/controller.log').exists(),'Unproven existing merged model'
            assert f'Merged {mode}' in (W/'startup-failure-1/controller.log').read_text()
            from safetensors import safe_open
            with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as tensors:
                assert len(list(tensors.keys()))>400
            print('Reusing completed merge',mode,flush=True)
            continue
        p=launch([PY,'-m','tau3_grpo.training.sft.merge','--base',str(R / 'code/models/Qwen3.5-4B'),
                  '--adapter',str(R / 'code/results/legacy/experiments/multicall-aa267bb/sft'/mode),'--output',str(M/('merged-'+mode))],
                 'merge-'+mode)
        assert p.wait()==0,f'Merge failed: {mode}'
        print('Merged',mode,flush=True)
    simulator=launch(serve(SIM,R / 'code/models/Qwen3.8-27B-AWQ-INT4','tau3-user',8300,.65,16384,False,True),
                     'simulator','simulator')
    ready(simulator,8300)
    outcomes={}
    for mode in ['off','on']:
        checkpoint=M/('merged-'+mode)
        policy=launch(serve(PY,checkpoint,'tau3-policy',8200,.8,24576,mode=='on'),'policy-'+mode,'policy')
        ready(policy,8200)
        att=W/('attestation-'+mode+'.json')
        write_service_attestation(checkpoint_path=str(checkpoint),served_model_name='tau3-policy',
                                  base_url='http://127.0.0.1:8200/v1',pid=policy.pid,output=att)
        probe=requests.post('http://127.0.0.1:8200/v1/chat/completions',json={
            'model':'tau3-policy','messages':[{'role':'user','content':'Reply with a short greeting.'}],
            'max_tokens':128,'temperature':0},timeout=120)
        probe.raise_for_status();(W/('service-probe-'+mode+'.json')).write_text(json.dumps(probe.json(),indent=2))
        cmd=[PY,'-m','tau3_grpo.evaluation.run','--target','selection','--checkpoint',str(checkpoint),
             '--policy-model','tau3-policy','--policy-base-url','http://127.0.0.1:8200/v1',
             '--user-model','tau3-user','--user-base-url','http://127.0.0.1:8300/v1',
             '--policy-attestation',str(att),'--seed','42','--data-seed','42','--trials','4',
             '--max-concurrency','2','--max-steps','30','--policy-temperature','0.4','--user-temperature','0',
             '--include-pass-hat','--output-dir',str(W/('results-'+mode))]
        print('Starting selection',mode,flush=True)
        evaluator=launch(cmd,'evaluation-'+mode)
        rc=evaluator.wait();(W/('evaluation-'+mode+'.exit')).write_text(str(rc));stop(policy)
        result_dir=W/('results-'+mode)
        if not (result_dir/'summary.json').exists():raise RuntimeError(f'No completed summary for {mode}')
        actual=json.loads((result_dir/'run.json').read_text())
        for key in ['planned','spec','benchmark_revision','endpoints']:
            assert actual[key]==reference[key],f'Evaluation differs from reference: {key}'
        rows=[json.loads(line) for name in ['trajectories.jsonl','errors.jsonl']
              for line in (result_dir/name).read_text().splitlines() if line.strip()]
        assert sorted((x['task_id'],x['trial'],x['seed']) for x in rows)==sorted((x['task_id'],x['trial'],x['seed']) for x in planned)
        outcomes[mode]={'exit_code':rc,'summary':json.loads((result_dir/'summary.json').read_text())}
        (W/'outcomes.json').write_text(json.dumps(outcomes,indent=2))
        print('Finished selection',mode,'exit',rc,'all 240 attempts accounted for',flush=True)
        # Completed budget/context errors remain explicitly invalid; still evaluate the other independent arm.
    (W/'controller.exit').write_text('0' if all(x['exit_code']==0 for x in outcomes.values()) else '1')
    print('Both arms completed; inspect outcomes.json for metric validity',flush=True)

try:main()
except BaseException as exc:
    (W/'controller.exit').write_text('1')
    print(type(exc).__name__,str(exc),flush=True)
    raise
finally:
    for process in reversed(processes):stop(process)
