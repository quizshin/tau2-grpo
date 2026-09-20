import json,torch
from transformers import AutoTokenizer,AutoModelForImageTextToText
root='/root/autodl-tmp/tau3-5xa800-20260911/model_store/Qwen3.5-4B'
t=AutoTokenizer.from_pretrained(root,local_files_only=True)
m=AutoModelForImageTextToText.from_pretrained(root,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa')
m.eval()
x=t('Hello',return_tensors='pt').to('cuda:0')
with torch.inference_mode():
 y=m(**x)
 assert torch.isfinite(y.logits).all()
print(json.dumps({'base_model_forward':True,'logits_shape':list(y.logits.shape),'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30}))
