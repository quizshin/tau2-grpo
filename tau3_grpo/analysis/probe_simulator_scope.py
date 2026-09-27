"""Independent authored simulator probes; no selection/final tasks or gold actions.

The automated gate checks explicit lexical obligations only. Saved replies require
semantic review; passing this small exposed probe set is not a drift-rate estimate.
"""
import argparse,json,re,time
from pathlib import Path
from tau3_grpo.envs.simulator_scope import scope_text
from tau3_grpo.utils.hashing import sha256_json

CASES=[
 dict(id='fixed_card',scenario='Your refund must go to your saved credit card ending in 6842. Do not use your other card ending in 3197.',
      history=[('user','Please refund my booking.')],agent='You have cards ending 3197 and 6842. Which should I use?',expect=r'6842',no_stop=True),
 dict(id='proposed_wrong_card',scenario='Pay only with credit_card_8401267. You do not authorize credit_card_9203158.',
      history=[('user','I would like to change my booking.')],agent='I can charge credit_card_9203158. Should I proceed?',expect=r'8401267',no_stop=True),
 dict(id='conditional_cabin',scenario='First ask to remove one passenger. If told that is impossible, ask to downgrade all existing passengers to basic economy. Do not cancel the booking.',
      history=[('user','Can you remove one person from my reservation?')],agent='I cannot change the passenger count. Is there anything else?',expect=r'basic\s*economy',no_stop=True),
 dict(id='conditional_not_yet',scenario='Ask to remove one passenger. Only after the agent refuses, request basic economy. Do not mention a downgrade before a refusal.',
      history=[('user','Can you remove a passenger?')],agent='Let me first check. What is your booking ID?',expect=r'not|don.t|know|remember|lookup|look up|find|check',forbid=r'basic\s*economy|downgrade|ABC123',no_stop=True),
 dict(id='unfinished_question',scenario='Request a cabin change to economy and ask why your outbound flight was cancelled. You have not asked about the cause yet.',
      history=[('user','Please change my cabin to economy.'),('assistant','The cabin is now economy.')],agent='Anything else before we finish?',expect=r'why|reason|cause',no_stop=True),
 dict(id='cabin_hard_preference',scenario='You want basic economy for every passenger. Business is not an allowed substitute.',
      history=[('user','I need to switch to basic economy.')],agent='I found business for all passengers. Confirm?',expect=r'basic\s*economy',no_stop=True),
 dict(id='identity_not_replace',scenario='Your name is Daniel Lee, the account holder. The booked passenger is Mira Chen. Ask only to verify Mira Chen spelling. Do not replace the passenger.',
      history=[('user','Please verify the spelling of Mira Chen on the ticket.')],agent='Should I replace Mira Chen with Daniel Lee?',expect=r'no|don.t|do not|keep|only|not',no_stop=True),
 dict(id='no_escalation',scenario='Ask for a baggage reduction. If refused, accept the refusal. Do not request a human, supervisor or transfer.',
      history=[('user','Can you remove a checked bag?')],agent='Baggage cannot be removed. Would you like me to transfer you to a human?',expect=r'no|not|fine|okay|ok|understand|thanks|thank',no_stop=False),
 dict(id='delayed_goal',scenario='First ask why a flight is delayed. After receiving an answer, ask for compensation. Do not forget the compensation request.',
      history=[('user','Why is my flight delayed?')],agent='The status is delayed, but the system does not provide the cause. Anything else?',expect=r'compensat|voucher|certificate',no_stop=True),
 dict(id='profile_fact',scenario='Your date of birth is recorded in your profile. You do not remember the date. Ask the agent to retrieve it rather than inventing a date.',
      history=[('user','Use the date of birth on my profile.')],agent='Is your date of birth 1993-04-05?',expect=r'profile|account|look|check|retrieve|verify',no_stop=True),
 dict(id='permitted_fallback',scenario='Prefer economy. If economy is unavailable, explicitly allow business. Do not demand economy after learning it is unavailable.',
      history=[('user','I prefer economy, but business is okay if economy is unavailable.')],agent='Economy is unavailable. Business is available. Do you want business?',expect=r'yes|business|proceed|confirm|sure',no_stop=True),
 dict(id='wrong_card_late',scenario='For any payment or refund, use the card ending in 5276, never 8043.',
      history=[('user','Please use my card ending 5276.'),('assistant','I will check availability.'),('user','Okay.')],agent='The change is ready. I will use the card ending 8043 unless you object.',expect=r'5276',no_stop=True),
]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--protocol',choices=['scenario_fidelity_v2','scenario_fidelity_v3'],default='scenario_fidelity_v3');p.add_argument('--temperature',type=float,default=0.7);a=p.parse_args()
 from tau2.user.user_simulator import UserSimulator
 from tau2.data_model.message import AssistantMessage,UserMessage
 a.output.mkdir(parents=True,exist_ok=False)
 config={'protocol':a.protocol,'temperature':a.temperature,'seeds':[42,43],'max_tokens':512,'cases':CASES,'scope_sha256':sha256_json(scope_text(a.protocol))}
 (a.output/'inputs.json').write_text(json.dumps(config,indent=2)+'\n')
 rows=[];started=time.time()
 with (a.output/'replies.jsonl').open('x') as log:
  for c in CASES:
   for seed in config['seeds']:
    sim=UserSimulator(llm='openai/Qwen/Qwen3.8-27B-AWQ-INT4',instructions=c['scenario']+'\n\n'+scope_text(config['protocol']),
       llm_args={'api_base':'http://127.0.0.1:8100/v1','api_key':'EMPTY','temperature':a.temperature,'max_tokens':512,'num_retries':0,'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}})
    sim.set_seed(seed)
    history=[(UserMessage if role=='user' else AssistantMessage)(role=role,content=t) for role,t in c['history']]
    message,_=sim.generate_next_message(AssistantMessage(role='assistant',content=c['agent']),sim.get_init_state(message_history=history))
    text=message.content or ''
    passed=bool(text.strip()) and (not c['expect'] or bool(re.search(c['expect'],text,re.I))) and (not c.get('forbid') or not re.search(c['forbid'],text,re.I)) and (not c['no_stop'] or not sim.is_stop(message))
    row={'id':c['id'],'seed':seed,'passed':bool(passed),'response':message.model_dump(mode='json')};rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
 receipt={'passed':all(r['passed'] for r in rows),'cases':len(CASES),'replies':len(rows),'passed_replies':sum(r['passed'] for r in rows),'input_sha256':sha256_json(config),'scope_sha256':config['scope_sha256'],'elapsed_seconds':time.time()-started,'scope':'authored independent probes; lexical gate, semantic review still required; no population drift estimate'}
 (a.output/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return 0 if receipt['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
