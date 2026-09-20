import copy, hashlib, json, subprocess
from pathlib import Path
import yaml
from transformers import AutoTokenizer
from tau3_grpo.training.sft.dataset import TrajectorySFTDataset
from tau3_grpo.prompts import prompt_provenance

R = Path('/root/autodl-tmp/tau3-5xa800-20260911')
C = R / 'code'
W = R / 'multicall-aa267bb'
base = R / 'model_store/Qwen3.5-4B'
tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
config = yaml.safe_load((C / 'configs/train/sft/qwen35_4b_lora_20260909.yaml').read_text())
config.pop('includes', None)
config.pop('launch', None)
config['model']['name_or_path'] = str(base)
tools_path = C / config['data']['tool_config']
tools = [x['tool_schema'] for x in yaml.safe_load(tools_path.read_text())['tools']]
report = {'prompt': prompt_provenance(), 'source_commit': subprocess.check_output(['git','-c',f'safe.directory={C}','-C',str(C),'rev-parse','HEAD'],text=True).strip(), 'gpu_assignment': {'reserved_other_thread':0,'off':1,'on':2}, 'arms':{}, 'sha256':{}}
for mode in ['off','on']:
    cfg = copy.deepcopy(config)
    opts = dict(enable_thinking=mode=='on',supervise_reasoning=mode=='on',preserve_historical_reasoning=mode=='on')
    cfg['data'].update(opts)
    cfg['output']['dir'] = str(W / 'sft' / mode)
    report['arms'][mode] = {}
    for split,n in [('train',45),('validation',5)]:
        path = C / cfg['data'][split+'_jsonl']
        cfg['data'][split+'_jsonl'] = str(path)
        dataset = TrajectorySFTDataset(path,tokenizer,tools=tools,max_length=24576,expected_size=n,**opts)
        originals = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        for original,effective in zip(originals,dataset.records):
            assert [m for m in original['messages'] if m['role']!='system'] == [m for m in effective['messages'] if m['role']!='system']
        report['arms'][mode][split] = {**dataset.token_stats(),'max_tokens':max(x['n_total_tokens'] for x in dataset.examples),'no_truncation':True}
        report['sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg['data']['tool_config'] = str(tools_path)
    (W / (mode+'.yaml')).write_text(yaml.safe_dump(cfg,sort_keys=False))
    report['sha256'][mode+'.yaml'] = hashlib.sha256((W/(mode+'.yaml')).read_bytes()).hexdigest()
for path in [tools_path,base/'config.json',base/'tokenizer.json',base/'chat_template.jinja']:
    if path.exists(): report['sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
(W/'preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
