"""Evidence joins must not manufacture training eligibility or lose exposure."""
import json
from copy import deepcopy

import pytest

from tau3_grpo.data.source_reuse import evidence_ledger
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def test_quality_merge_rejects_changed_message_or_mask():
    from tau3_grpo.data.finalize_staged_sft import reviewed_quality_rows

    row = dialogue()
    decision = dict(id='one', decision='keep', reason='reviewed',
                    exact_messages_mask_sha256=sha256_json([row['messages'], [1]]))
    assert reviewed_quality_rows([row], [decision]) == ([row], [])
    changed = deepcopy(row)
    changed['messages'][1]['content'] = 'Different supervised answer'
    with pytest.raises(ValueError, match='identity mismatch'):
        reviewed_quality_rows([changed], [decision])
    changed = deepcopy(row)
    changed['supervision']['message_indices'] = []
    with pytest.raises(ValueError, match='identity mismatch'):
        reviewed_quality_rows([changed], [decision])


def test_quality_merge_requires_complete_explicit_decisions():
    from tau3_grpo.data.finalize_staged_sft import reviewed_quality_rows

    row = dialogue()
    with pytest.raises(ValueError, match='exact unique input'):
        reviewed_quality_rows([row], [])
    decision = dict(id='one', decision='unknown', reason='unresolved',
                    exact_messages_mask_sha256=sha256_json([row['messages'], [1]]))
    with pytest.raises(ValueError, match='Explicit keep/drop'):
        reviewed_quality_rows([row], [decision])
    decision['decision'] = 'drop'
    assert reviewed_quality_rows([row], [decision]) == ([], [row])


def test_quality_merge_blocks_heldout_user_in_tool_receipt():
    from tau3_grpo.data.finalize_staged_sft import check_quality_split

    train, dev = dialogue('train'), dialogue('dev')
    train['messages'].append(dict(role='tool', name='get_reservation_details',
                                 content=json.dumps(dict(user_id='protected_user'))))
    dev['messages'][0]['content'] = 'Another task'
    dev['metadata']['entity_group'] = 'protected_user'
    with pytest.raises(ValueError, match='visible user collision'):
        check_quality_split([train], [dev])


def test_quality_merge_blocks_exact_duplicates():
    from tau3_grpo.data.finalize_staged_sft import check_quality_split

    with pytest.raises(ValueError, match='Exact dialogue/mask duplicate'):
        check_quality_split([dialogue('train')], [dialogue('dev')])


def test_quality_merge_blocks_source_only_user_collision():
    from tau3_grpo.data.finalize_staged_sft import check_quality_split

    train, dev = dialogue('train'), dialogue('dev')
    dev['messages'][0]['content'] = 'Different task'
    train['metadata']['source_user_id'] = 'source_only_user'
    dev['metadata']['entity_group'] = 'source_only_user'
    with pytest.raises(ValueError, match='visible user collision'):
        check_quality_split([train], [dev])


def fixture_config(tmp_path, rows):
    datasets = []
    for index, (row, split) in enumerate(rows):
        path = tmp_path / f'{index}.jsonl'
        path.write_text(json.dumps(row) + '\n')
        datasets.append(dict(name=str(index), path=str(path), sha256=sha256_file(path),
                             locked_split=split, exposure='candidate_only', exposure_basis='test'))
    decisions = tmp_path / 'decisions.jsonl'
    decisions.write_text('')
    return dict(datasets=datasets, decisions=str(decisions), source_review_records=str(tmp_path),
                accepted_training_cohorts=['0'])


def dialogue(sid='one', kind=None):
    metadata = dict(source_dialog_id=sid)
    if kind:
        metadata['repair_kind'] = kind
    return dict(messages=[dict(role='user', content='Read only.'),
                          dict(role='assistant', content='Done.')],
                supervision=dict(message_indices=[1]), metadata=metadata)


def test_alias_dedup_keeps_all_memberships_and_split_collision(tmp_path):
    a, b = dialogue('old', 'lookup'), dialogue('new', 'lookup')
    result = evidence_ledger(fixture_config(tmp_path, [(a, 'train'), (b, 'validation')]), [])
    assert len(result['records']) == 1
    record = result['records'][0]
    assert record['source_aliases'] == ['new', 'old']
    assert len(record['memberships']) == 2
    assert record['historical_split_locks'] == ['train', 'validation']
    assert 'exact_content_train_dev_collision' in record['training_candidate_blockers']
    assert len(result['template_overlap']) == 1
    assert result['manifest']['splits'] == {'train': [], 'validation': []}


def test_same_source_changed_content_kept_separate(tmp_path):
    a, b = dialogue(), dialogue()
    b['messages'][1]['content'] = 'Different answer.'
    result = evidence_ledger(fixture_config(tmp_path, [(a, 'train'), (b, 'train')]), [])
    assert len(result['records']) == 2
    assert len(result['source_content_variants']) == 1


def test_masks_are_part_of_dedup_identity(tmp_path):
    a = dialogue()
    a['messages'].append(dict(role='assistant', content='Second target.'))
    b = deepcopy(a)
    b['supervision']['message_indices'] = [1, 2]
    result = evidence_ledger(fixture_config(tmp_path, [(a, 'train'), (b, 'train')]), [])
    assert len(result['records']) == 2
    assert len({r['messages_sha256'] for r in result['records']}) == 1


def test_stale_file_and_review_rejected(tmp_path):
    row = dialogue()
    config = fixture_config(tmp_path, [(row, 'train')])
    review = dict(source_id='one', messages_sha256='stale')
    with pytest.raises(ValueError, match='Review does not match'):
        evidence_ledger(config, [review])
    config['datasets'][0]['sha256'] = 'stale'
    with pytest.raises(ValueError, match='Input changed'):
        evidence_ledger(config, [])


def test_reviewed_difficulty_does_not_bypass_family_gate(tmp_path):
    row = dialogue()
    review = dict(source_id='one', messages_sha256=sha256_json(row['messages']),
                  verdict='pass_visible_evidence', evidence=[dict(message_indices=[0, 1])],
                  difficulty='easy', needs_replanning=False, needs_dependent_subgoals=False,
                  family='lookup')
    result = evidence_ledger(fixture_config(tmp_path, [(row, 'train')]), [review])
    assert len(result['previews']['foundation']) == 1
    assert result['labels'][0]['label_status'] == 'reviewed'
    assert result['manifest']['splits']['train'] == []
    assert result['records'][0]['training_ready'] is True
    assert result['records'][0]['independent_dev_ready'] is False
    assert len(result['reviewed_training_manifest']['splits']['train']) == 1


def test_generic_template_overlap_does_not_discard_training_candidate(tmp_path):
    a, b = dialogue('train', 'lookup'), dialogue('dev', 'lookup')
    b['messages'][0]['content'] = 'A different read-only task.'
    result = evidence_ledger(fixture_config(tmp_path, [(a, 'train'), (b, 'validation')]), [])
    assert len(result['template_overlap']) == 1
    assert sum(r['training_ready'] for r in result['records']) == 1
    assert not any(r['independent_dev_ready'] for r in result['records'])


def test_invalid_supervision_rejected(tmp_path):
    row = dialogue()
    row['supervision']['message_indices'] = [0]
    with pytest.raises(ValueError, match='Non-assistant supervision'):
        evidence_ledger(fixture_config(tmp_path, [(row, 'train')]), [])


def test_new_cohort_requires_matching_semantic_pass(tmp_path):
    row=dialogue()
    config=fixture_config(tmp_path,[(row,'train')])
    config['accepted_training_cohorts']=[]
    config['new_review_required_cohorts']=['0']
    assert not evidence_ledger(config,[])['records'][0]['training_ready']
    review=dict(source_id='one',messages_sha256=sha256_json(row['messages']),
                verdict='pass_visible_evidence',evidence=[dict(message_indices=[0,1])])
    result=evidence_ledger(config,[review])['records'][0]
    assert result['training_ready']
    assert result['acceptance_basis']=='new_full_dialogue_semantic_review'
    review['verdict']='hold_new_semantic_issue'
    assert not evidence_ledger(config,[review])['records'][0]['training_ready']


def test_legacy_label_hash_uses_historical_unicode_serialization(tmp_path):
    from tau3_grpo.analysis.sft_coldstart_audit import digest
    row=dialogue();row['messages'][0]['content']='查航班'
    config=fixture_config(tmp_path,[(row,'train')])
    path=tmp_path/'legacy.jsonl'
    label=dict(source_id='one',messages_sha256=digest(row['messages']),label_status='pending')
    path.write_text(json.dumps(label)+'\n');config['legacy_labels']=str(path)
    result=evidence_ledger(config,[])['records'][0]
    assert result['inherited_curriculum_labels']==[label]
    assert result['messages_sha256']!=label['messages_sha256']


def test_fresh_token_counts_keep_provenance_and_require_dataset_identity(tmp_path):
    row=dialogue();config=fixture_config(tmp_path,[(row,'train')])
    path=tmp_path/'tokens.json'
    report=dict(source_sha256=config['datasets'][0]['sha256'],
                rows=[dict(id='one',assistant_tokens=2,total_tokens=10)])
    path.write_text(json.dumps(report))
    config['token_sources']=[dict(dataset='0',path=str(path),selector=['rows'],id_field='id',
                                 count_basis='fresh_local_tokenizer_render')]
    record=evidence_ledger(config,[])['records'][0]
    assert record['historical_render_token_counts']==[]
    assert record['render_token_counts'][0]['count_basis']=='fresh_local_tokenizer_render'
    report['source_sha256']='stale';path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Token count source identity mismatch'):
        evidence_ledger(config,[])
