"""Rebuild the assistant-reviewed diagnostic fixture from existing selection logs.

Labels were assigned by reading prefixes, not rewards or anchor predictions.
This is a purposive diagnostic set, not independent human annotation/test data.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

EXPECTED_E2 = 'c561ac450bd899eacbcbc44bd11a501263ae945db0285beb1790d03c36400a92'
# task, trials, relation, reason. All natural pairs are before any tool call.
NATURAL = [
 ('1059',0,3,'merge','同样要求将现有预订从商务舱降至经济舱；省钱动机不改变操作范围。'),
 ('1060',0,2,'merge','相同的商务舱到经济舱降舱请求与省钱动机。'),
 ('1060',0,3,'merge','相同舱位变更，后者只是重述当前商务舱。'),
 ('1106',0,2,'merge','同为 HKEG34 改日期、加两件托运行李、先查询完整预订。'),
 ('1106',2,3,'merge','订单、日期变更、两件行李及先查询要求完全相同。'),
 ('126',0,1,'merge','HKEG34、DEN-LAS、次日、最佳直飞、8am-9pm 相同；机场缩写与城市名称同义。'),
 ('126',1,2,'merge','同一订单、航线、次日直飞和时间窗口。'),
 ('14',0,2,'merge','同为 HKEG34 先改日期再加两件行李；请求帮助/查询不增加修改权限。'),
 ('14',2,3,'merge','订单及改日期再加两件行李的目标一致。'),
 ('1143',1,2,'merge','相同身份和订单、出发航班取消/返程延误事实、原因查询与补偿请求；情绪措辞不同。'),
 ('1143',2,3,'merge','相同身份、订单、航线日期、扰动事实和补偿诉求。'),
 ('162',0,2,'merge','同一姓名、ORD-ATL、May25、三次重复预订；保留一个取消两个，overlapping 不增加独立选择约束。'),
 ('1035',0,2,'separate','明确 business→economy 的请求与仅说修改预订，已知操作目标不同。'),
 ('1059',0,1,'separate','前者明确两种舱位，后者仅表达省钱，不能凭隐藏任务补齐。'),
 ('1059',0,2,'separate','后者额外暴露 round-trip，已知预订范围不同。'),
 ('1073',0,2,'separate','后者明确只保留成年人，前者未说明保留哪些乘客。'),
 ('1073',1,3,'separate','前者还有取消行程问题；后者只说 May22 的航班变更。'),
 ('109',0,1,'separate','明确升级舱位与泛化修改预订的目标不同。'),
 ('109',0,2,'separate','前者提供姓名，后者提供 basic economy→economy；身份及舱位信息均不同。'),
 ('1106',0,1,'separate','后者提供姓名但没有改日期/两件行李，不能因订单相同就合并。'),
 ('1113',0,1,'separate','前者明确保留晚班且取消晨班，后者没有保留晚班约束。'),
 ('112',0,2,'separate','后者增加两名乘客这一已知事实。'),
 ('1143',0,1,'separate','前者明确 Gold 身份，涉及补偿资格，不能忽略。'),
 ('14',0,1,'separate','同一订单，但后者尚未要求加两件行李。'),
 ('1035',1,2,'uncertain','同一 user ID，但一侧额外提供姓名；是否可将姓名作为冗余信息需身份知识规则。'),
 ('1113',1,3,'uncertain','一侧明确 tomorrow；缺少可靠参考时间，不能自动判断与 May20 完全等价。'),
 ('126',0,3,'uncertain','一侧明确 departure window，另一侧只说 between 8am-9pm，需核验是否约束出发或整个行程。'),
 ('162',0,3,'uncertain','一侧为明确 ORD，另一侧仅 Chicago，多机场城市不能未经核验视为同一出发机场。'),
]


def clean(messages):
    return [{k:deepcopy(m[k]) for k in ('role','content','tool_calls','id','tool_call_id','error') if k in m} for m in messages]


def state(messages,task='synthetic_task',db='CONTROLLED_EQUAL_DB',budget=12):
    return {'task_id':task,'db_hash':db,'remaining_assistant_turns':budget,'messages':clean(messages)}


def msg(role,content):return {'role':role,'content':content}


def build(source):
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_E2:raise ValueError('Source changed; review labels against the new source first')
    rows={}
    for line in source.open():
        if line.strip():
            row=json.loads(line);rows[(row['task_id'],row['trial'])]=row
    pairs=[]
    def add(kind,relation,reason,a,b,refs=None):
        pairs.append({'id':f'{kind}-{sum(p["kind"]==kind for p in pairs)+1:02d}',
                      'kind':kind,'relation':relation,'rationale':reason,
                      'source_refs':refs or [],'a':a,'b':b})
    def prefix(task,trial,end):
        full=rows[(task,trial)]['simulation']['messages']
        subset=full[:end+1]
        assert all(c['name'] in {'get_user_details','get_reservation_details','search_direct_flight','search_onestop_flight','list_all_airports','get_flight_status'} for m in subset for c in m.get('tool_calls') or [])
        return state(subset,task,'MATCHED_DB_ASSUMPTION:'+task),{'arm':'e2','task_id':task,'trial':trial,'through_message_index':end,'prefix_sha256':hashlib.sha256(json.dumps(clean(subset),sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    for task,ta,tb,label,reason in NATURAL:
        a,ra=prefix('airline_'+task,ta,1);b,rb=prefix('airline_'+task,tb,1)
        assert len(a['messages'])==len(b['messages'])==2
        assert a['messages'][0]==b['messages'][0]
        assert a['messages'][1]['role']==b['messages'][1]['role']=='user'
        add('natural',label,reason,a,b,[ra,rb])
    specs=[
      ('airline_1035',0,9,'Yes, check the economy price for the flight.',
       'Yes, please proceed with the downgrade now.',
       '查询许可同义改写；改写未增加修改许可。','查询价格许可与立即修改舱位许可必须区分。'),
      ('airline_1060',1,13,None,
       'Yes, proceed with the downgrade and refund to the original card ending in 7388.',
       '仅将开头 Wait 换为 Please pause，保留全部退款卡更正条件。','更正退款卡并暂停，与接受原卡退款不同。'),
      ('airline_1106',2,12,None,
       'Yes, I approve the flight change now and accept the proposed refund to credit_card_1955700.',
       '仅将 Hold on a second 换为 Please wait a moment，全部身份、行李及付款条件原样保留。','有条件的未来批准与立即无条件批准不同。'),
    ]
    for task,trial,end,paraphrase,contrast,yes_reason,no_reason in specs:
        a,ref=prefix(task,trial,end)
        if paraphrase is None:
            original=a['messages'][-1]['content']
            paraphrase=original.replace('Wait,','Please pause,',1) if task=='airline_1060' else original.replace('Hold on a second.','Please wait a moment.',1)
            assert paraphrase!=original
        for relation,text,reason in [('merge',paraphrase,yes_reason),('separate',contrast,no_reason)]:
            b=deepcopy(a);b['messages'][-1]['content']=text
            add('real_prefix_counterfactual',relation,reason,a,b,[ref,{'modified_last_user_message':True,'observed_trajectory':False}])
    proposal='Cancel reservation ABC123 for a $100 refund to card card_1. Do you confirm?'
    a=state([msg('assistant',proposal),msg('user','Yes, please proceed.')])
    b=deepcopy(a);b['messages'][-1]['content']='I approve.'
    add('controlled','merge','相同完整方案下的两种明确批准表述。',a,b)
    for old,new,label in [('card_1','card_2','退款目的卡'),('$100','$200','退款金额'),('ABC123','ABC124','订单对象')]:
        b=deepcopy(a);b['messages'][0]['content']=proposal.replace(old,new)
        add('controlled','separate',label+'不同，同一句 yes 不能合并。',a,b)
    for text,reason in [('Yes, but only if there is no fee.','带条件批准不能当作无条件批准。'),('No, do not proceed.','拒绝与批准不同。'),('I revoke my approval.','撤回历史批准必须更新当前状态。')]:
        left=deepcopy(a);left['messages'] += [msg('assistant',proposal),msg('user','Yes, please proceed.')]
        b=deepcopy(left);b['messages'][-1]['content']=text
        add('controlled','separate',reason,left,b)
    read=state([{'role':'assistant','content':None,'tool_calls':[{'id':'call_1','name':'get_reservation_details','arguments':{'reservation_id':'ABC123'}}]},
                {'role':'tool','id':'call_1','content':'{"reservation_id":"ABC123","cabin":"business"}','error':False}])
    b=deepcopy(read);b['messages'][0]['tool_calls'][0]['id']='other_id';b['messages'][1]['id']='other_id'
    add('controlled','merge','工具调用 ID 仅用于配对，不是环境状态。',read,b)
    b=deepcopy(read);b['messages'][1]['content']='{ "cabin": "business", "reservation_id": "ABC123" }'
    add('controlled','merge','JSON 键顺序与空白不改变读到的事实。',read,b)
    b=deepcopy(read);b['messages'][1]['content']='{"reservation_id":"ABC123","cabin":"economy"}'
    add('controlled','separate','DB 相同假设下，两条轨迹已知读事实不同。',read,b)
    for field,value,reason in [('db_hash','DIFFERENT_DB','数据库状态不同。'),('task_id','other_task','任务不同，不属于同一 episode 比较范围。')]:
        b=deepcopy(a);b[field]=value;add('controlled','separate',reason,a,b)
    b=deepcopy(a);b['remaining_assistant_turns']=1
    add('external_budget','uncertain','同样对话但剩余行动预算为 12 与 1；有限时域价值可能不同，但论文允许跨时间重复状态。是否按预算拆组是建模选择，不判作算法错误。',a,b)
    return {'schema_version':1,'annotation':{'author':'Codex assistant, context-reviewed','independent_human_labels':False,'selection':'purposive, before reading predictions; correlated pairs, no population accuracy claim','target':'candidate visible decision-state abstraction; not proof of full policy/user-simulator Markov-state equivalence','db_scope':'natural prefixes precede tools; later real prefixes contain read tools only; matched DB supplied, intermediate DB NOT replayed','outcomes_used_for_labels':False},'sources':{'e2/trajectories.jsonl':digest},'pairs':pairs}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=build(args.source);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    from collections import Counter
    print(json.dumps({'pairs':len(result['pairs']),'by_kind_relation':dict(Counter(p['kind']+'/'+p['relation'] for p in result['pairs']))},ensure_ascii=False))
