import json,os,signal,subprocess,time
from pathlib import Path
import requests
from tau3_grpo.evaluation.service_attestation import write_service_attestation
R=Path('/root/autodl-tmp/tau3-5xa800-20260911');W=R/'selection-thinking-gpu01'
PY=str(R/'venvs/qwen35/bin/python');SIM=str(R/'venvs/qwen38-sim/bin/python')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']='4'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['VLLM_NO_USAGE_STATS']='1'
os.environ['VLLM_CACHE_ROOT']=str(W/'cache')
processes=[]
def interrupted(signum,frame):raise KeyboardInterrupt('Controller interrupted')
signal.signal(signal.SIGTERM,interrupted)
def launch(cmd,name):
 log=(W/(name+'.log')).open('w')
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1' if name=='simulator' else '0'
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=R/'code');log.close();processes.append(p);return p

def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def ready(p,port):
 end=time.monotonic()+600
 while time.monotonic()<end:
  if p.poll() is not None:raise RuntimeError(f'service {port} exited {p.returncode}')
  try:
   if requests.get(f'http://127.0.0.1:{port}/health',timeout=3).status_code==200:return
  except requests.RequestException:pass
  time.sleep(3)
 raise TimeoutError(f'service {port} not ready')

def serve(python,model,name,port,memory,context,thinking,sim=False):
 cmd=[python,'-m','vllm.entrypoints.cli.main','serve',str(model),'--served-model-name',name,'--host','127.0.0.1','--port',str(port),'--tensor-parallel-size','1','--dtype','bfloat16','--max-model-len',str(context),'--max-num-seqs','2','--gpu-memory-utilization',str(memory),'--enforce-eager','--language-model-only','--default-chat-template-kwargs',json.dumps({'enable_thinking':thinking}),'--override-generation-config',json.dumps({'max_new_tokens':2048})]
 if sim:cmd+=['--quantization','compressed-tensors']
 else:cmd+=['--enable-auto-tool-choice','--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3']
 return cmd

if (W/'controller.started').exists():raise RuntimeError('Controller already started; inspect state before resuming')
(W/'controller.started').write_text(str(time.time()))
plan={'policy_gpu':0,'simulator_gpu':1,'methods':['on'],'tasks':60,'trials':4,'concurrency':2,'policy_temperature':.4,'user_temperature':0,'max_new_tokens_per_turn':2048,'policy_context':24576,'user_context':16384,'max_steps':30,'thinking_on_changes_inference_mode':True}
(W/'plan.json').write_text(json.dumps(plan,indent=2))
try:
 sim=launch(serve(SIM,R/'model_store/Qwen3.8-27B-AWQ-INT4','tau3-user',8100,.65,16384,False,True),'simulator');ready(sim,8100)
 for method in ['on']:
  checkpoint=R/'selection-single-gpu'/('merged-'+method)
  p=launch(serve(PY,checkpoint,'tau3-policy',8000,.80,24576,method=='on'),'policy-'+method);ready(p,8000)
  att=W/('attestation-'+method+'.json')
  write_service_attestation(checkpoint_path=str(checkpoint),served_model_name='tau3-policy',base_url='http://127.0.0.1:8000/v1',pid=p.pid,output=att)
  probe=requests.post('http://127.0.0.1:8000/v1/chat/completions',json={'model':'tau3-policy','messages':[{'role':'user','content':'Reply with a short greeting.'}],'max_tokens':128,'temperature':0},timeout=120);probe.raise_for_status()
  (W/('service-probe-'+method+'.json')).write_text(json.dumps(probe.json(),indent=2))
  cmd=[PY,'-m','tau3_grpo.evaluation.run','--target','selection','--checkpoint',str(checkpoint),'--policy-model','tau3-policy','--policy-base-url','http://127.0.0.1:8000/v1','--user-model','tau3-user','--user-base-url','http://127.0.0.1:8100/v1','--policy-attestation',str(att),'--trials','4','--max-concurrency','2','--max-steps','30','--policy-temperature','0.4','--user-temperature','0','--include-pass-hat','--output-dir',str(W/('results-'+method))]
  print('Starting selection',method,flush=True);evaluation=launch(cmd,'evaluation-'+method);rc=evaluation.wait();(W/('evaluation-'+method+'.exit')).write_text(str(rc));stop(p)
  if rc:raise RuntimeError(f'evaluation {method} exited {rc}; results preserved')
 (W/'controller.exit').write_text('0');print('Thinking selection run completed',flush=True)
except BaseException as exc:
 (W/'controller.exit').write_text('1');print(type(exc).__name__,str(exc),flush=True);raise
finally:
 for p in reversed(processes):stop(p)
