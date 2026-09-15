import json,os,yaml
from pathlib import Path
from functools import partial
import torch
from transformers import AutoTokenizer,TrainingArguments
from peft import PeftModel
from tau3_grpo.models.compat import load_policy_model
from tau3_grpo.training.sft.dataset import TrajectorySFTDataset,collate_fn_padding
from tau3_grpo.training.sft.trainer import Qwen35SFTTrainer
r=Path(os.environ['TAU3_ROOT']);e=r/'thinking-experiment'
t=AutoTokenizer.from_pretrained(r/'model_store/Qwen3.5-4B',local_files_only=True)
t.padding_side='right'
tools=[x['tool_schema'] for x in yaml.safe_load((r/'code/configs/envs/tool_config.yaml').read_text())['tools']]
d=TrajectorySFTDataset(r/'code/data/sft/airline_sft_validation_seed42.jsonl',t,tools=tools,max_length=24576,expected_size=5)
b=load_policy_model(str(r/'model_store/Qwen3.5-4B'),torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False)
m=PeftModel.from_pretrained(b,r/'sft-comparison/lora-thinking',is_trainable=False)
a=TrainingArguments(output_dir=str(e/'evaluation'),per_device_eval_batch_size=1,bf16=True,report_to='none',remove_unused_columns=False)
tr=Qwen35SFTTrainer(model=m,args=a,eval_dataset=d,data_collator=partial(collate_fn_padding,pad_token_id=t.pad_token_id),processing_class=t)
metrics=tr.evaluate()
old=json.loads((r/'sft-comparison/lora/train_summary.json').read_text())['validation_metrics']['eval_loss']
result={'target':'same_answer_only_validation_as_baseline','thinking_trained_answer_only_loss':metrics['eval_loss'],'baseline_answer_only_loss':old,'validation_dialogues':5,'not_task_success_evaluation':True}
(e/'answer-only-comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
