"""Entity-disjoint capability train/dev candidates using real training DBs.

Families are locked before dialogue generation. Source selection uses only train,
old dev and exposed selection protections. These are procedural development probes,
not an unseen distribution, independent templates, or final benchmark tasks.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from tau3_grpo.analysis.capability_distribution import task_feature
from tau3_grpo.data.grounded_gap_pilot import Builder, build_case, feasible, raw_database, read_rows
from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.schema import ArealTaskRecord
from tau3_grpo.data.sft_expansion import audit_tool_calls
from tau3_grpo.data.staged_sft import ordered_tool_receipts
from tau3_grpo.envs.adapter import adapt_record
from tau3_grpo.paths import AREAL_DB_ROOT
from tau3_grpo.utils.hashing import sha256_file, sha256_json

KINDS=('baggage_free','baggage_paid','cabin_basic_upgrade','cabin_economy_upgrade',
       'cancel_allowed','cancel_denied','compensation_1','passenger_correction',
       'flight_change','flight_refusal','airport_lookup','connection_lookup','book_direct','transfer_flown')
BASE_KINDS=set(KINDS[:7])


def user_ids(rows):
    ids=set()
    for row in rows:
        for m in row['messages']:
            for c in m.get('tool_calls') or []:
                a=c.get('function',c).get('arguments',{})
                if isinstance(a,str):a=json.loads(a)
                if a.get('user_id'):ids.add(a['user_id'])
    return ids


def choose_flight(db,r,*,book=False):
    # Single segment reduces ambiguity; existing source supplies complex bookings.
    if len(r['flights'])!=1 or r['flight_type']!='one_way':return None
    old=r['flights'][0];cabin='economy' if book else r['cabin'];n=len(r['passengers'])
    options=[]
    for fid,f in db['flights'].items():
        if (f['origin'],f['destination'])!=(r['origin'],r['destination']):continue
        for date,state in f['dates'].items():
            if date<=max('2024-05-15',old['date']) or state.get('status')!='available':continue
            if state['available_seats'][cabin]<n:continue
            options.append((date,state['prices'][cabin],fid))
    return min(options) if options else None


def suitable(db,rid,kind):
    if kind in BASE_KINDS:return feasible(db,rid,kind)
    r=db['reservations'].get(rid)
    if not r or r.get('status')=='cancelled' or not r['passengers']:return False
    if any(f['flight_number'] not in db['flights'] or f['date'] not in db['flights'][f['flight_number']]['dates'] for f in r['flights']):return False
    u=db['users'][r['user_id']];future=all(f['date']>'2024-05-15' for f in r['flights'])
    cards=any(p['source']=='credit_card' for p in u['payment_methods'].values())
    if kind=='passenger_correction':return future
    if kind=='flight_change':return future and cards and r['cabin']!='basic_economy' and choose_flight(db,r) is not None
    if kind=='flight_refusal':return future and r['cabin']=='basic_economy'
    if kind=='book_direct':return future and cards and len(r['passengers'])<=5 and choose_flight(db,r,book=True) is not None
    if kind=='transfer_flown':return any(db['flights'][f['flight_number']]['dates'][f['date']].get('status') in ('flying','landed') for f in r['flights'])
    if kind=='connection_lookup':return future and len(r['flights'])==2 and r['flight_type']=='one_way'
    return future


def build_extra(entry,rid,kind):
    adapted=adapt_record(ArealTaskRecord.model_validate(entry.task))
    if adapted.db_file_hash!=entry.db_hash:raise ValueError('Source DB identity changed')
    b=Builder(adapted,entry,rid,kind);db=b.before;r=db['reservations'][rid];uid=r['user_id'];u=db['users'][uid]
    b.expected=copy.deepcopy(db);er=b.expected['reservations'][rid];n=len(r['passengers'])
    cards=sorted(k for k,v in u['payment_methods'].items() if v['source']=='credit_card')
    checks=['real_tools','independent_replay','explicit_target_and_protected_state_checked']
    if kind=='passenger_correction':
        old=r['passengers'][0];new={**old,'first_name':old['first_name']+'a'}
        reason=f'Correct one passenger first name on {rid}, preserving count and all other details.'
        b.say('user',f'My user ID is {uid}. On reservation {rid}, please correct passenger {old["first_name"]} {old["last_name"]}, born {old["dob"]}: the first name must be {new["first_name"]}. Keep their last name, birth date, every other passenger and all travel details unchanged.')
        b.tool('get_reservation_details',{'reservation_id':rid})
        b.say('assistant',f'I found {n} passengers. I will change only the first name of {old["first_name"]} {old["last_name"]}, born {old["dob"]}, to {new["first_name"]}. Passenger count, other passengers, flights and payment details will stay unchanged. May I proceed?')
        b.say('user','Yes, make only that first-name correction.')
        er['passengers'][0]=new
        b.tool('update_reservation_passengers',{'reservation_id':rid,'passengers':er['passengers']})
        b.say('assistant',f'The first name is now {new["first_name"]}. The other passenger details and the reservation are otherwise unchanged.')
        checks+=['explicit_identity_from_user','preserved_passenger_multiset_except_requested_field']
    elif kind=='flight_refusal':
        reason=f'Request a date change without changing basic-economy cabin; accept policy refusal and preserve {rid}.'
        b.say('user',f'My user ID is {uid}. For reservation {rid}, can I move my flights to a later date while keeping basic economy? I do not want a cabin upgrade, cancellation or a new booking. If the policy forbids it, explain and leave my booking unchanged.')
        b.tool('get_reservation_details',{'reservation_id':rid})
        b.say('assistant','This reservation is basic economy. The policy does not allow changing its flights. A cabin-only change is a separate permitted action before departure, but you declined an upgrade. I have left your reservation unchanged.')
        b.say('user','Understood; keep my original booking. That is all.')
        b.say('assistant','Your original booking remains unchanged. No cancellation or new booking was made.')
        checks+=['correct_refusal','no_unrequested_workaround']
    elif kind=='transfer_flown':
        reason=f'Cancellation assistance for already-flown reservation {rid}; transfer with no cancellation.'
        b.say('user',f'My user ID is {uid}. I would like to cancel reservation {rid} because my plans changed. Please check whether that is still possible. If this is outside your capabilities, please transfer me.')
        b.tool('get_reservation_details',{'reservation_id':rid})
        for f in r['flights']:b.tool('get_flight_status',{'flight_number':f['flight_number'],'date':f['date']})
        b.say('assistant','At least part of this reservation has already flown. I cannot cancel it with these tools under the policy, so I will transfer you for assistance.')
        b.tool('transfer_to_human_agents',{'summary':f'User {uid} requests cancellation of {rid} for changed plans; at least one segment has already flown. No cancellation or refund performed.'})
        b.say('assistant','YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.')
        checks+=['verified_flown_status','no_illegal_cancellation','transfer_tool_before_required_message']
    elif kind=='airport_lookup':
        reason='Ask for airport codes for a city pair before choosing a flight; do not book.'
        names={'SFO':'San Francisco','JFK':'New York','LAX':'Los Angeles','ORD':'Chicago','DFW':'Dallas','DEN':'Denver','SEA':'Seattle','ATL':'Atlanta','MIA':'Miami','BOS':'Boston','PHX':'Phoenix','IAH':'Houston','LAS':'Las Vegas','MCO':'Orlando','EWR':'Newark','CLT':'Charlotte','MSP':'Minneapolis','DTW':'Detroit','PHL':'Philadelphia','LGA':'LaGuardia'}
        a,z=r['origin'],r['destination']
        b.say('user',f'Which airport codes does your service list for {names[a]} and {names[z]}? I only need the codes; do not search for tickets or create a booking.')
        listing=b.tool('list_all_airports',{})
        if not all(any(x['iata']==code for x in listing) for code in (a,z)):raise ValueError('Airport code missing')
        b.say('assistant',f'{names[a]} is listed as {a}, and {names[z]} as {z}. No booking has been made.')
        checks+=['read_only_fact_grounding']
    elif kind=='connection_lookup':
        date=r['flights'][0]['date'];origin=r['origin'];destination=r['destination']
        reason=f'Inspect one-stop options from {origin} to {destination} on {date}; no booking.'
        b.say('user',f'Please check one-stop flights from {origin} to {destination} on {date} for {n} passengers in economy. Tell me one available itinerary and its total fare. I am only checking options; do not book anything.')
        options=b.tool('search_onestop_flight',{'origin':origin,'destination':destination,'date':date})
        options=[o for o in options if all(f['available_seats']['economy']>=n for f in o)]
        if not options:raise ValueError('No verified one-stop option')
        one=min(options,key=lambda o:(sum(f['prices']['economy'] for f in o),str(o)))
        total=sum(f['prices']['economy'] for f in one)*n
        b.tool('calculate',{'expression':f'({one[0]["prices"]["economy"]}+{one[1]["prices"]["economy"]})*{n}'})
        b.say('assistant',f'One available economy itinerary is {one[0]["flight_number"]} on {one[0]["date"]}, {origin} to {one[0]["destination"]}, then {one[1]["flight_number"]} on {one[1]["date"]} to {destination}. The combined fare is ${total} for {n} passengers, before any optional extras. I have not booked it.')
        checks+=['native_connection_results','seats_checked','no_unrequested_booking']
    elif kind in ('flight_change','book_direct'):
        book=kind=='book_direct';date,price,fid=choose_flight(db,r,book=book);cabin='economy' if book else r['cabin'];card=cards[-1]
        newflight=db['flights'][fid];old=r['flights'][0];total=price*n
        if book:
            reason=f'Book one-way {r["origin"]} to {r["destination"]} on {date}, specified flight {fid}, no baggage or insurance.'
            b.say('user',f'My user ID is {uid}. I want a NEW one-way economy booking from {r["origin"]} to {r["destination"]} on {date}, flight {fid}, for these passengers: {json.dumps(r["passengers"])}. I need no checked bags. Leave all existing reservations unchanged.')
            b.tool('get_user_details',{'user_id':uid})
            options=b.tool('search_direct_flight',{'origin':r['origin'],'destination':r['destination'],'date':date})
            if not any(f['flight_number']==fid for f in options):raise ValueError('Selected flight missing')
            b.say('assistant','Would you like travel insurance? It costs $30 per passenger and covers cancellation for health or weather reasons.')
            b.say('user',f'No insurance. Use my stored credit card {card}.')
            b.tool('calculate',{'expression':f'{price}*{n}'})
            b.say('assistant',f'Please confirm the new one-way economy booking: {fid} on {date}, {r["origin"]} to {r["destination"]}, {n} passengers {json.dumps(r["passengers"])}, no checked bags and no insurance. The total is ${total}, paid using {card}. Shall I book it?')
            b.say('user','Yes, book exactly those details at that total.')
            args=dict(user_id=uid,origin=r['origin'],destination=r['destination'],flight_type='one_way',cabin=cabin,
                flights=[dict(flight_number=fid,date=date)],passengers=r['passengers'],payment_methods=[dict(payment_id=card,amount=total)],
                total_baggages=0,nonfree_baggages=0,insurance='no')
            receipt=b.tool('book_reservation',args);newid=receipt['reservation_id']
            if newid in db['reservations']:raise ValueError('New reservation ID overwrote old state')
            b.expected['reservations'][newid]=dict(reservation_id=newid,user_id=uid,origin=r['origin'],destination=r['destination'],flight_type='one_way',cabin=cabin,
                flights=[dict(flight_number=fid,date=date,origin=r['origin'],destination=r['destination'],price=price)],passengers=copy.deepcopy(r['passengers']),
                payment_history=[dict(payment_id=card,amount=total)],created_at='2024-05-15T15:00:00',total_baggages=0,nonfree_baggages=0,insurance='no',status=None)
            b.expected['users'][uid]['reservations'].append(newid)
            b.expected['flights'][fid]['dates'][date]['available_seats'][cabin]-=n
            b.say('assistant',f'Booked successfully: reservation {newid}, flight {fid} on {date}. The ${total} payment used {card}. There is no insurance or checked baggage, and your existing reservations are unchanged.')
            checks+=['all_booking_fields_confirmed','new_reservation_payment_and_seats_verified']
        else:
            delta=total-old['price']*n
            reason=f'Change only {rid} to specified flight {fid} on {date}, same route/cabin/passengers, using {card}.'
            b.say('user',f'My user ID is {uid}. Please change reservation {rid} to flight {fid} on {date}. Keep the same route, cabin, passengers, bags and insurance. I want to use {card} for the fare difference. Check the price before updating.')
            b.tool('get_reservation_details',{'reservation_id':rid});b.tool('get_user_details',{'user_id':uid})
            b.tool('search_direct_flight',{'origin':r['origin'],'destination':r['destination'],'date':date})
            b.tool('calculate',{'expression':f'({price}-{old["price"]})*{n}'})
            description=f'an additional charge of ${delta}' if delta>=0 else f'a refund of ${-delta}'
            b.say('assistant',f'The requested flight {fid} on {date} is available in {cabin} for all {n} passengers. The old airfare was ${old["price"]*n}, and the new airfare is ${total}, giving {description} using {card}. All other booking details stay unchanged. May I update reservation {rid}?')
            b.say('user',f'Yes, change to {fid} on {date} with that fare difference and payment method.')
            er['flights']=[dict(flight_number=fid,date=date,origin=r['origin'],destination=r['destination'],price=price)]
            if delta:er['payment_history'].append(dict(payment_id=card,amount=delta))
            b.tool('update_reservation_flights',dict(reservation_id=rid,cabin=cabin,flights=[dict(flight_number=fid,date=date)],payment_id=card))
            b.say('assistant',f'Reservation {rid} is now on {fid} on {date}. The update produced {description} using {card}; the other booking details are unchanged.')
            checks+=['single_flight_target_verified','fare_delta_from_observed_old_price','profile_payment_verified']
    else:raise ValueError(kind)
    return b.finish(reason,checks)


def prepare(output):
    old_train=user_ids(read_rows('data/sft/staged_v2_A100_20260925/train.jsonl'))
    old_dev=user_ids(read_rows('data/sft/staged_v2_A100_20260925/validation.jsonl'))
    selection_users=set()
    for e in read_manifest('data/manifests/areal_airline_selection_seed42.jsonl'):
        record=ArealTaskRecord.model_validate(e.task);db=raw_database(str(record.resolve_db_path(AREAL_DB_ROOT)))
        text=json.dumps(record.user_scenario)+json.dumps(record.evaluation_criteria)
        selection_users.update(uid for uid in db['users'] if uid in text)
    protected=old_dev|selection_users;pool={}
    for entry in read_manifest('data/manifests/areal_airline_train_seed42.jsonl'):
        if entry.split!='train' or entry.task.get('initial_state'):continue
        record=ArealTaskRecord.model_validate(entry.task);db=raw_database(str(record.resolve_db_path(AREAL_DB_ROOT)))
        for rid in task_feature(entry.task)['reservation_ids']:
            r=db['reservations'].get(rid)
            if not r or r['user_id'] in protected:continue
            pool.setdefault((entry.db_hash,rid),dict(task_id=entry.task_id,rid=rid,user_id=r['user_id'],db_hash=entry.db_hash))
    entries={e.task_id:e for e in read_manifest('data/manifests/areal_airline_train_seed42.jsonl')}
    eligible={}
    for kind in KINDS:
        items=[]
        for x in pool.values():
            e=entries[x['task_id']];db=raw_database(str(ArealTaskRecord.model_validate(e.task).resolve_db_path(AREAL_DB_ROOT)))
            if suitable(db,x['rid'],kind):items.append(x)
        eligible[kind]=sorted(items,key=lambda x:sha256_json([42,kind,x['user_id'],x['rid']]))
    # Freeze one unseen-user probe per branch BEFORE choosing training variants.
    dev=[];dev_users=set();used=set()
    for kind in sorted(KINDS,key=lambda k:len({x['user_id'] for x in eligible[k] if x['user_id'] not in old_train})):
        options=[x for x in eligible[kind] if x['user_id'] not in old_train and x['user_id'] not in dev_users]
        if not options:raise ValueError('Insufficient distinct unseen-user probe family: '+kind)
        x=options[0];dev.append(dict(kind=kind,split='validation',**x));dev_users.add(x['user_id']);used.add(x['rid'])
    train=[];counts=Counter()
    for kind in sorted(KINDS,key=lambda k:len(eligible[k])):
        selected_users=set()
        for x in sorted(eligible[kind],key=lambda x:(counts[x['user_id']],sha256_json([42,kind,x['rid']]))):
            if x['user_id'] in dev_users|selected_users or x['rid'] in used:continue
            train.append(dict(kind=kind,split='train',**x));used.add(x['rid']);selected_users.add(x['user_id']);counts[x['user_id']]+=1
            if len(selected_users)==3:break
        if not selected_users:raise ValueError('No protected training family for: '+kind)
    plan=dict(version='grounded_entity_split_v1',requested_train_per_kind=3,actual_train_counts=dict(Counter(x['kind'] for x in train)),train=train,validation=dev,heldout_users=sorted(dev_users),
        protected_users=sorted(protected),historical_train_users=sorted(old_train),source_sha256=sha256_file(__file__),
        base_builder_sha256=sha256_file(Path(__file__).with_name('grounded_gap_pilot.py')),
        final_or_reserve_read=False,notes=['Entity split; templates are shared, not independent task-family generalization.',
          'New source records using heldout users must be excluded from training before freeze.'])
    output.mkdir(parents=True,exist_ok=False);(output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    return plan,entries


def run(args):
    import yaml
    plan,entries=prepare(args.output)
    schemas=[x['tool_schema'] for x in yaml.safe_load(Path('configs/envs/tool_config.yaml').read_text())['tools']]
    results=[];errors=[]
    for item in plan['train']+plan['validation']:
        try:
            builder=build_case if item['kind'] in BASE_KINDS else build_extra
            row=builder(entries[item['task_id']],item['rid'],item['kind'])
            if row is None:raise ValueError('Frozen selection failed live feasibility')
            row['metadata'].update(source_dialog_id=f"repair_{item['kind']}_{item['task_id']}_{item['rid']}",
                split=item['split'],entity_group=item['user_id'],construction_kind=item['kind'],
                plan_sha256=sha256_file(args.output/'plan.json'),status='executed_candidate_needs_final_review',
                difficulty={'level':'unrated','basis':'branch rubric pending semantic review'})
            failures=audit_tool_calls(row,schemas)+ordered_tool_receipts(row)
            if failures:raise ValueError(failures)
            with (args.output/(item['split']+'.jsonl')).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            results.append(item)
        except Exception as exc:errors.append(dict(**item,error=str(exc)[:500]))
        print(json.dumps({'done':len(results),'errors':len(errors),'kind':item['kind'],'split':item['split']}),flush=True)
    report=dict(planned=len(plan['train'])+len(plan['validation']),completed=len(results),splits=dict(Counter(x['split'] for x in results)),errors=errors,
        counts=dict(Counter(x['kind'] for x in results)),gpu_hours=0,new_api_calls=0,training_frozen=False,
        input_plan_sha256=sha256_file(args.output/'plan.json'))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
