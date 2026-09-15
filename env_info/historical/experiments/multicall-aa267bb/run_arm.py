import datetime, json, os, subprocess, sys
from pathlib import Path

R = Path('/root/autodl-tmp/tau3-5xa800-20260911')
C = R/'code'
W = R/'multicall-aa267bb'
mode = sys.argv[1]
gpu = {'off':3,'on':2}[mode]
uuid = {'off':'GPU-ff283ad1-81a9-85e0-014f-7a28f5c8aed5','on':'GPU-63d13965-cbe4-900f-b057-25d52f285032'}[mode]
audit=json.loads((W/'preflight.json').read_text())
assert audit['source_commit']=='aa267bba7f581a607f5ed4fed88668f9ecda9b55'
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
if uuid in active: raise RuntimeError(f'GPU{gpu} already occupied; refusing to share it')
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(C))
output = W/'sft'/mode
if output.exists(): raise RuntimeError('Run directory exists; refusing to overwrite')
start=W/(mode+'.started.json')
with start.open('x') as f: json.dump({'pid':os.getpid(),'gpu':gpu,'gpu_uuid':uuid,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':audit['source_commit']},f,indent=2)
cmd=[sys.executable,'-m','tau3_grpo.training.sft.train','--config',str(W/(mode+'.yaml')),'--report-to','swanlab','--run-name',f'multicall-aa267bb-sft-{mode}-seed42']
(W/(mode+'.command.json')).write_text(json.dumps(cmd))
with (W/(mode+'.train.log')).open('w') as log:
    rc=subprocess.call(cmd,env=env,cwd=C,stdout=log,stderr=subprocess.STDOUT)
(W/(mode+'.train.exit')).write_text(str(rc))
if rc==0:
    cmd=[sys.executable,'-m','tau3_grpo.training.sft.evaluate','--config',str(W/(mode+'.yaml')),'--model',str(output),'--base-model',str(R/'model_store/Qwen3.5-4B'),'--output',str(W/(mode+'.answer-only.json'))]
    with (W/(mode+'.eval.log')).open('w') as log:
        rc=subprocess.call(cmd,env=env,cwd=C,stdout=log,stderr=subprocess.STDOUT)
(W/(mode+'.completion.exit')).write_text(str(rc))
sys.exit(rc)
