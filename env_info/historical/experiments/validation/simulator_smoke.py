import json,os,signal,subprocess,time
from pathlib import Path
import requests
r=Path('/root/autodl-tmp/tau3-5xa800-20260911')
cmd='source '+str(r/'activate.sh')+'; export TAU3_USER_ENFORCE_EAGER=1; export TAU3_USER_MAX_NUM_SEQS=2; bash "$TAU3_ROOT/code/scripts/serve/simulator_qwen38.sh"'
with (r/'validation/simulator-service.log').open('w') as log:
 p=subprocess.Popen(['bash','-c',cmd],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  deadline=time.monotonic()+420
  while time.monotonic()<deadline:
   if p.poll() is not None:raise RuntimeError('Simulator process exited '+str(p.returncode))
   try:
    response=requests.get('http://127.0.0.1:8100/health',timeout=2)
    if response.status_code==200:break
   except requests.RequestException:pass
   time.sleep(3)
  else:raise TimeoutError('Simulator readiness timeout')
  response=requests.post('http://127.0.0.1:8100/v1/chat/completions',json={'model':'Qwen/Qwen3.8-27B-AWQ-INT4','messages':[{'role':'user','content':'Reply with a short greeting.'}],'max_tokens':16,'temperature':0},timeout=90)
  response.raise_for_status();data=response.json();assert data.get('choices'),data
  result={'health':True,'chat_completion':True,'usage':data.get('usage'),'gpu':4,'enforce_eager':True}
  (r/'validation/simulator-smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
