import json
import pytest
from test_qwen35 import tokenizer, MESSAGES, TOOLS
from tau3_grpo.training.sft.evaluate import validation_dataset, baseline_loss
from tau3_grpo.training.sft.dataset import build_supervised_example
from tau3_grpo.prompts import prepare_agent_messages


def test_answer_only_ignores_training_thinking_flags(tmp_path):
    tok = tokenizer('Qwen3.5-4B')
    messages = [dict(m, reasoning='REASONING_NOT_A_TARGET') if m['role']=='assistant' else m for m in MESSAGES]
    data = tmp_path/'validation.jsonl'
    data.write_text('\n'.join(json.dumps({'messages':messages}) for _ in range(5)))
    tools = tmp_path/'tools.yaml'
    tools.write_text(json.dumps({'tools':[{'tool_schema':x} for x in TOOLS]}))
    cfg={'data':dict(validation_jsonl=str(data),tool_config=str(tools),max_length=24576,
                     enable_thinking=True,supervise_reasoning=True,preserve_historical_reasoning=True)}
    ds, provenance=validation_dataset(cfg,tok)
    expected=build_supervised_example(prepare_agent_messages(messages),tok,tools=TOOLS,max_length=24576)
    assert ds.examples[0]['labels']==expected['labels']
    assert ds.examples[0]['input_ids']==expected['input_ids']
    off_cfg={'data':{**cfg['data'],'enable_thinking':False,
                     'supervise_reasoning':False,'preserve_historical_reasoning':False}}
    off_ds, off_provenance=validation_dataset(off_cfg,tok)
    assert ds.examples==off_ds.examples
    assert provenance==off_provenance
    assert 'REASONING_NOT_A_TARGET' not in tok.decode(
        [value for value in ds.examples[0]['labels'] if value != -100])
    assert len(provenance['validation_source_sha256'])==64
    summary={'validation':ds.token_stats(),'enable_thinking':False,'validation_metrics':{'eval_loss':.42}}
    assert baseline_loss(summary,ds.token_stats())==.42
    with pytest.raises(ValueError):baseline_loss(dict(summary,enable_thinking=True),ds.token_stats())
    with pytest.raises(ValueError):baseline_loss(summary,{})
