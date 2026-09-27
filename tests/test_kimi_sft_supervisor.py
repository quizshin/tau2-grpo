import json

import pytest

from tau3_grpo.analysis.kimi_sft_supervisor import select_pending
from tau3_grpo.utils.hashing import sha256_file


def test_existing_unknown_attempt_is_never_selected(tmp_path):
    (tmp_path / 'candidate1').mkdir()
    plan = {'candidates': [{'candidate_id': 'candidate1'}]}
    assert select_pending(plan, tmp_path, set()) == []
    with pytest.raises(ValueError, match='Frozen Codex200'):
        select_pending(plan, tmp_path, {'candidate1'})


def test_source_changes_and_incomplete_generation_block_paid_review(tmp_path):
    candidate = tmp_path / 'source.json'
    rubric = tmp_path / 'rubric.jsonl'
    candidate.write_text(json.dumps({'candidate_id': 'candidate1', 'task': {'task_hash': 'h'},
                                     'status': 'failed'}))
    rubric.write_text('{}\n')
    row = {'candidate_id': 'candidate1', 'task_hash': 'h', 'candidate_file': str(candidate),
           'candidate_sha256': sha256_file(candidate), 'rubrics_file': str(rubric),
           'rubrics_sha256': sha256_file(rubric)}
    plan = {'candidates': [row]}
    with pytest.raises(ValueError, match='Incomplete generation'):
        select_pending(plan, tmp_path, set())
    candidate.write_text('{}')
    with pytest.raises(ValueError, match='identity changed'):
        select_pending(plan, tmp_path, set())
