"""Thinking SFT: preserve supplied reasoning at all turns, supervise assistant only."""
import copy
from tau3_grpo.models.qwen35_template import token_ids

def templates(tokenizer):
    native = tokenizer.chat_template
    condition = 'loop.index0 > ns.last_query_index'
    assert native.count(condition) == 1
    native = native.replace(condition, 'true')
    old = "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}"
    new = "{{- '<|im_start|>' + message.role + '\\n' }}{%- generation %}{{- '<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}{%- endgeneration %}"
    assert native.count(old) == 1
    marked = native.replace(old,new)
    marker='{%- elif message.role == "assistant" %}'
    before,_,rest=marked.partition(marker)
    body,end,after=rest.partition('{%- elif message.role == "tool" %}')
    tool='{%- if message.tool_calls and'
    eos="{{- '<|im_end|>\\n' }}"
    assert body.count(tool)==1 and body.count(eos)==1
    body=body.replace(tool,'{%- generation %}'+tool).replace(eos,eos+'{%- endgeneration %}')
    return native,before+marker+body+end+after

def build(messages,tokenizer,*,tools=None,max_length=24576):
    messages=copy.deepcopy(messages)
    for m in messages:
        if m['role']=='assistant':m['reasoning_content']=m.get('reasoning','')
    native,marked=templates(tokenizer)
    kwargs=dict(tools=tools,tokenize=True,add_generation_prompt=False,enable_thinking=True)
    ids=token_ids(tokenizer.apply_chat_template(messages,chat_template=native,**kwargs))
    encoded=tokenizer.apply_chat_template(messages,chat_template=marked,return_dict=True,return_assistant_tokens_mask=True,**kwargs)
    assert ids==token_ids(encoded)
    mask=encoded['assistant_masks']
    if mask and isinstance(mask[0],list):mask=mask[0]
    assert len(ids)==len(mask) and any(mask)
    if len(ids)>max_length:raise ValueError('Thinking dialogue exceeds max_length; no truncation allowed')
    return dict(input_ids=ids,labels=[x if m else -100 for x,m in zip(ids,mask)],attention_mask=[1]*len(ids),n_total_tokens=len(ids),n_label_tokens=sum(mask))
