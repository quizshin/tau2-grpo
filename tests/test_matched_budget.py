import json

import pytest

from tau3_grpo.integrations.matched_budget import MatchedBudget


@pytest.mark.parametrize('observed,target', [(1, 10), (9, 10), (10, 10), (17, 20), (29, 30), (30, 30), (31, 40)])
def test_threshold_rounds_up_to_one_fixed_target(observed, target, tmp_path):
    budget = MatchedBudget(100, 21600, state_path=tmp_path / 'budget.json')
    assert not budget.observe(observed, now=21699)
    assert budget.target_step is None
    assert budget.observe(observed, now=21700) == (observed == target)
    assert budget.target_step == target
    for step in range(observed + 1, target):
        assert not budget.observe(step, now=30000)
        assert budget.target_step == target
    assert budget.observe(target, now=40000)
    state = json.loads((tmp_path / 'budget.json').read_text())
    assert state['target_step'] == target
    assert state['observed_step_at_threshold'] == observed


def test_only_e0_discovers_and_intervals_must_match(monkeypatch, tmp_path):
    monkeypatch.delenv('TAU3_E0_DISCOVERY_SECONDS', raising=False)
    assert MatchedBudget.from_environment(save_freq=10, test_freq=10) is None
    monkeypatch.setenv('TAU3_E0_DISCOVERY_SECONDS', '21600')
    monkeypatch.setenv('TAU3_BUDGET_STARTED_AT', '100')
    monkeypatch.setenv('TAU3_BUDGET_STATE_PATH', str(tmp_path / 'state.json'))
    monkeypatch.setenv('TAU3_GRPO_ARM', 'e3')
    with pytest.raises(ValueError, match='Only E0'):
        MatchedBudget.from_environment(save_freq=10, test_freq=10)
    monkeypatch.setenv('TAU3_GRPO_ARM', 'e0')
    with pytest.raises(ValueError, match='intervals'):
        MatchedBudget.from_environment(save_freq=10, test_freq=5)
    assert MatchedBudget.from_environment(save_freq=10, test_freq=10).interval == 10


def test_discovery_ceiling_still_selects_a_shared_target():
    budget = MatchedBudget(100, 21600)
    assert budget.observe(100, now=200, ceiling=100)
    assert budget.target_step == 100


@pytest.mark.parametrize('threshold_step,target', [(10, 10), (17, 20), (29, 30)])
def test_real_trainer_control_blocks_save_validate_then_stop(threshold_step, target, monkeypatch):
    """Execute actual fit control blocks with compute/save/eval stubs (no GPU)."""
    import ast
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).parents[1] / 'verl/verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(path.read_text())
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    fit = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    nodes = sorted(ast.walk(fit), key=lambda n: getattr(n, 'lineno', 0))
    target_assignment = next(n for n in nodes if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'tau3_target' for t in n.targets))
    last_assignment = next(n for n in nodes if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'is_last_step' for t in n.targets))
    def find_if(prefix):
        return next(n for n in nodes if isinstance(n, ast.If) and ast.unparse(n.test).startswith(prefix))
    save = find_if('self.config.trainer.save_freq > 0')
    validate = find_if('self.config.trainer.test_freq > 0')
    observe = find_if('tau3_matched_budget is not None')
    assert save.lineno < validate.lineno < observe.lineno
    module = ast.fix_missing_locations(ast.Module(body=[target_assignment, last_assignment, save, validate, observe], type_ignores=[]))
    code = compile(module, str(path), 'exec')
    events = []
    fake = SimpleNamespace(config=SimpleNamespace(trainer=SimpleNamespace(save_freq=10, test_freq=10)),
                           total_training_steps=100)
    fake._save_checkpoint = lambda: events.append(('save', fake.global_steps))
    def evaluate():
        events.append(('eval', fake.global_steps))
        return {'score': fake.global_steps}
    fake._validate = evaluate
    current_time = [0]
    monkeypatch.setattr('tau3_grpo.integrations.matched_budget.time.time', lambda: current_time[0])
    env = dict(self=fake, tau3_matched_budget=MatchedBudget(0, 21600),
               esi_close_to_expiration=False, marked_timer=lambda *a, **kw: nullcontext(),
               timing_raw={}, metrics={}, progress_bar=SimpleNamespace(total=100), last_val_metrics=None)
    for step in range(1, 101):
        fake.global_steps = step
        current_time[0] = 21600 if step >= threshold_step else 100
        exec(code, env)
        if env['is_last_step']:
            break
    assert fake.global_steps == target
    assert events == [(kind, step) for step in range(10, target + 1, 10) for kind in ('save', 'eval')]
    assert env['last_val_metrics'] == {'score': target}
