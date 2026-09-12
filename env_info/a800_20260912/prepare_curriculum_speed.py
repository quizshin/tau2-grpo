"""Audit frozen curriculum assets and resolve schedules; never launch training."""
import hashlib
import json
import os
from pathlib import Path
from string import Template
from collections import Counter

from tau3_grpo.paths import CODE_ROOT, DATA_ROOT
from tau3_grpo.launch import load_config, prepare
from tau3_grpo.experiments.prepare import prepare_experiment_inputs
from tau3_grpo.experiments.manifest import read_manifest, flatten_schedule, verify_extension


R=Path(os.environ['TAU3_ROOT'])
W=R/'runs/curriculum-speed-20260912'
C=CODE_ROOT/'results/analysis/rl_curriculum_20260912'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def resolved(stage='smoke'):
    profiles={'smoke':'pilot','pilot':'pilot','extend15':'extend15','extend20':'extend20'}
    profile=CODE_ROOT/f'configs/train/rl/qwen35_4b_full_a800_curriculum40_fast_{profiles[stage]}_20260912.yaml'
    env=os.environ.copy()
    env['CODE_ROOT']=str(CODE_ROOT)
    # A stale custom parquet would bypass the selected 40-task manifest.
    env.pop('TRAIN_PARQUET',None)
    for key,value in load_config(profile)['launch']['environment'].items():
        env[key]=Template(str(value)).substitute(env)
    extra=[]
    if stage=='smoke':
        env.update(TOTAL_UPDATES='1',RESULTS_DIR=str(W/'online-smoke'),
            SWANLAB_EXPERIMENT_NAME='engineering-E0-c40-g8x8-u1-paddingfix')
        # Engineering measurement: no 50 GiB checkpoint; pilot retains full state.
        extra=['trainer.resume_mode=disable','trainer.save_freq=-1']
    result=Path(env['RESULTS_DIR'])
    env.update(TOOL_CONFIG=str(result/'tool_config.yaml'),
        SWANLAB_LOG_DIR=str(result/'swanlog'),
        TAU3_GRPO_DEBUG_BATCH_DIR=str(result/'update-batches'),
        VERL_QWEN35_WEIGHT_AUDIT_DIR=str(result/'weight-audits'))
    command,env,snapshot=prepare('rl',profile,'e0',42,extra,env)
    assert int(env['GROUP_SIZE'])*int(env['GROUPS_PER_UPDATE'])==64
    assert env['MODEL_PATH']==str(R/'checkpoints/sft-merged/new-off')
    assert Path(env['TRAIN_MANIFEST_DIR']).resolve()==(C/'manifests').resolve()
    return command,env,snapshot


def main():
    W.mkdir(parents=True,exist_ok=True)
    candidates=read_jsonl(C/'manifests/areal_airline_train_seed42.jsonl')
    parent={x['task_id']:x for x in read_jsonl(DATA_ROOT/'manifests/areal_airline_train_seed42.jsonl')}
    selection={x['task_id'] for x in read_jsonl(DATA_ROOT/'manifests/areal_airline_selection_seed42.jsonl')}
    assert len(candidates)==len({x['task_id'] for x in candidates})==40
    assert not ({x['task_id'] for x in candidates}&selection)
    assert all(x==parent[x['task_id']] for x in candidates)
    audit=json.loads((C/'audit.json').read_text())
    for db,digest in audit['db_hashes'].items():
        assert sha(DATA_ROOT/'raw/areal_tau2'/db)==digest,db
    actual_sft=read_jsonl(DATA_ROOT/'sft/airline_sft_train_seed42.jsonl')
    actual_ids={x['metadata']['source_dialog_id'] for x in actual_sft}
    preflight=json.loads((R/'experiments/multicall-aa267bb/preflight.json').read_text())
    recorded=[v for k,v in preflight['sha256'].items() if k.endswith('airline_sft_train_seed42.jsonl')]
    assert recorded==[sha(DATA_ROOT/'sft/airline_sft_train_seed42.jsonl')]
    # Original classification remains unchanged. Confirm each selected SFT link
    # is present in the ACTUAL new-off training set, rather than only the old audit.
    rows=[x for x in read_jsonl(C/'all_200_classified.jsonl') if x['selected']]
    links=[x['best_clean_match'] for x in rows]
    link_ids=[x['id'] if 'id' in x else x['sft_id'] for x in links]
    assert all(x in actual_ids for x in link_ids)
    schedules={}
    for updates in (1,10,15,20,50):
        out=W/f'plan-{updates}'
        prepare_experiment_inputs(arm='e0',seed=42,data_seed=42,group_size=8,
            groups_per_update=8,total_updates=updates,anchor_mode='structured',
            manifest_dir=C/'manifests',output_dir=out)
        schedules[updates]=read_manifest(out)
    verify_extension(schedules[1],schedules[10]);verify_extension(schedules[10],schedules[50])
    verify_extension(schedules[10],schedules[15]);verify_extension(schedules[15],schedules[20])
    assert set(Counter(flatten_schedule(schedules[10].schedule)).values())=={2}
    assert set(Counter(flatten_schedule(schedules[50].schedule)).values())=={10}
    assert set(Counter(flatten_schedule(schedules[15].schedule)).values())=={3}
    assert set(Counter(flatten_schedule(schedules[20].schedule)).values())=={4}
    report={'task_count':40,'candidate_manifest_sha256':sha(C/'manifests/areal_airline_train_seed42.jsonl'),
        'all_candidates_identical_to_frozen_train':True,'selection_id_overlap':0,
        'db_hashes_verified':audit['db_hashes'],'actual_sft_input_sha256':recorded[0],
        'all_40_best_sft_links_present_in_actual_new_off_input':True,
        'smoke_first_8_tasks':flatten_schedule(schedules[1].schedule),
        'planned_trajectory_budgets':{'smoke':64,'pilot':640,'extend15':960,'extend20':1280,'extend50':3200},
        'schedule_prefix_1_10_50_verified':True,
        'schedule_prefix_10_15_20_verified':True,
        'scope':'Identity, hash and schedule checks; not semantic solvability or reward proof.'}
    for stage in ('smoke','pilot','extend15','extend20'):
        _,_,snapshot=resolved(stage)
        (W/f'{stage}.launch.json').write_text(json.dumps(snapshot,indent=2))
    (W/'data-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
