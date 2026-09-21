"""Versioned evaluation inputs, exercised without model or service calls."""

import json
from types import SimpleNamespace

import pytest

from tau3_grpo.data.opening import DEFAULT_OPENING, initial_user_message
from tau3_grpo.evaluation import runtime
from tau3_grpo.evaluation.harness import (
    CONTROL_V2,
    INPUTS_V3,
    LEGACY,
    protocol_metadata,
    request_args,
)
from tau3_grpo.evaluation.runtime import Endpoint, EvalSpec


@pytest.mark.parametrize('instructions,expected', [
    ({'reason_for_call': 'Please change my flight', 'task_instructions': 'SECRET'}, 'Please change my flight'),
    ({'reason_for_call': '', 'known_info': 'SECRET'}, DEFAULT_OPENING),
    ({'reason_for_call': None}, DEFAULT_OPENING),
    ('hidden scenario', DEFAULT_OPENING),
    ({'reason_for_call': 123}, '123'),
])
def test_opening_preserves_parquet_rule_without_hidden_information(instructions, expected):
    task = {'user_scenario': {'instructions': instructions}, 'evaluation_criteria': {'secret': True}}
    before = json.dumps(task)
    assert initial_user_message(task) == expected
    assert json.dumps(task) == before


@pytest.mark.parametrize('protocol', [LEGACY, CONTROL_V2])
def test_old_request_defaults_unchanged(protocol):
    endpoint = Endpoint('policy', 'http://unused', temperature=1.0)
    assert request_args(protocol, endpoint, role='policy') == endpoint.llm_args()
    assert request_args(protocol, endpoint, role='user') == endpoint.llm_args()
    assert 'initial_user_message' not in protocol_metadata(protocol)


def test_v3_request_limits_are_explicit_and_temperatures_are_independent():
    endpoint = Endpoint('policy', 'http://unused')
    policy = request_args(INPUTS_V3, endpoint, role='policy')
    assert policy['max_tokens'] == 1024
    assert policy['extra_body'] == {'chat_template_kwargs': {'enable_thinking': False}}
    assert 'max_tokens' not in endpoint.llm_args()
    assert request_args(INPUTS_V3, endpoint, role='user')['temperature'] == endpoint.temperature
    user = request_args(INPUTS_V3, Endpoint('user', 'http://unused', temperature=0), role='user')
    assert user['temperature'] == 0
    assert 'max_tokens' not in user  # formal simulator has no per-request override
    assert protocol_metadata(INPUTS_V3)['token_context_equivalence'] == 'not_established'


def test_v3_preflight_rejects_before_loading_final(monkeypatch, tmp_path):
    target, temperature = 'tau3-final', 1
    monkeypatch.setattr(runtime, '_selection_jobs', lambda *a: pytest.fail('loaded tasks'))
    monkeypatch.setattr(runtime, '_official_jobs', lambda *a: pytest.fail('opened final'))
    with pytest.raises(ValueError):
        runtime.run_evaluation(spec=EvalSpec(target=target, harness_protocol=INPUTS_V3),
            policy=Endpoint('p', 'unused'), user=Endpoint('u', 'unused', temperature=temperature),
            output_dir=tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.fixture
def selection_record():
    from tau3_grpo.data.manifest import read_manifest
    from tau3_grpo.paths import MANIFEST_ROOT

    path = MANIFEST_ROOT / 'areal_airline_selection_seed42.jsonl'
    if not path.exists():
        pytest.skip('selection manifest unavailable')
    return read_manifest(path)[0]


def test_v3_native_loop_seeds_both_histories_and_executes_real_tools(monkeypatch, selection_record):
    from tau2.agent import llm_agent
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.user import user_simulator
    from tau2.utils.llm_utils import to_litellm_messages

    from tau3_grpo.data.parquet_builder import build_row
    from tau3_grpo.envs.orchestrator import TrainingInputOrchestrator
    from tau3_grpo.paths import TAU2_BENCH_ROOT

    job = next(runtime._selection_jobs([selection_record], 1, 42))
    original = job['task'].model_dump(mode='json')
    policy_calls, user_calls, instances = [], [], []
    native_init = TrainingInputOrchestrator.initialize

    def initialize(self):
        native_init(self)
        instances.append(self)
        assert self.real_user_messages == 1 and self.observation_batches == 0
        assert len(self.user_state.messages) == 1
        assert self.agent_state.messages == []
        assert self.user_state.messages[0].content == job['initial_user_message']

    monkeypatch.setattr(TrainingInputOrchestrator, 'initialize', initialize)

    def agent_generate(**kwargs):
        policy_calls.append({**kwargs, 'messages': to_litellm_messages(kwargs['messages'])})
        if len(policy_calls) == 1:
            return AssistantMessage(role='assistant', tool_calls=[
                ToolCall(id='calc1', name='calculate', arguments={'expression': '1+1'}),
                ToolCall(id='calc2', name='calculate', arguments={'expression': '2+2'}),
            ])
        return AssistantMessage(role='assistant', content='Done.')

    def user_generate(**kwargs):
        user_calls.append({**kwargs, 'messages': to_litellm_messages(kwargs['messages'])})
        return UserMessage(role='user', content='###STOP###')

    monkeypatch.setattr(llm_agent, 'generate', agent_generate)
    monkeypatch.setattr(user_simulator, 'generate', user_generate)
    simulation = runtime._run_one(task=job['task'], db_path=job['db_path'],
        policy=Endpoint('policy', 'http://unused'), user=Endpoint('user', 'http://unused', temperature=0),
        seed=42, max_steps=30, max_errors=10, harness_protocol=INPUTS_V3,
        initial_user_message=job['initial_user_message'])
    row = build_row(selection_record, policy=TAU2_BENCH_ROOT.joinpath(
        'data/tau2/domains/airline/policy.md').read_text(), split='selection')
    assert policy_calls[0]['messages'] == row['prompt']
    assert len(policy_calls) == 2 and len(user_calls) == 1
    assert [m['content'] for m in policy_calls[1]['messages'] if m['role'] == 'tool'] == ['2.0', '4.0']
    for call in policy_calls:
        assert call['max_tokens'] == 1024
        assert call['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert user_calls[0]['temperature'] == 0
    assert user_calls[0]['messages'][1:] == [
        {'role': 'assistant', 'content': job['initial_user_message'], 'tool_calls': None},
        {'role': 'user', 'content': 'Done.'},
    ]
    assert simulation.termination_reason.value == 'user_stop'
    assert instances[0].real_user_messages == 2 and instances[0].assistant_generations == 2
    assert job['task'].model_dump(mode='json') == original
    assert simulation.info['initial_user_message_sha256']
    assert simulation.info['harness_protocol']['version'] == INPUTS_V3


def test_v3_rejects_history_and_blank_opening_before_requests(selection_record):
    from tau2.data_model.message import UserMessage
    from tau2.data_model.tasks import InitialState

    from tau3_grpo.evaluation.harness import validate_opening

    job = next(runtime._selection_jobs([selection_record], 1, 42))
    with pytest.raises(ValueError, match='nonempty'):
        validate_opening(job['task'], '   ')
    task = job['task'].model_copy(update={'initial_state': InitialState(
        message_history=[UserMessage(role='user', content='existing')])})
    with pytest.raises(ValueError, match='fresh'):
        validate_opening(task, 'hello')


@pytest.mark.parametrize('protocol', [LEGACY, CONTROL_V2, INPUTS_V3])
def test_cli_temperature_defaults_and_final_guard(monkeypatch, capsys, protocol):
    from tau3_grpo.evaluation import run

    monkeypatch.setattr(run, 'read_manifest', lambda path: [SimpleNamespace(task_id='a')])
    flags = ['--target', 'selection', '--checkpoint', '/merged', '--dry-run', '--harness-protocol', protocol]
    assert run.main(flags) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['user_temperature'] == 0.7
    assert payload['policy_temperature'] == 0.7
    if protocol == INPUTS_V3:
        assert payload['harness_protocol']['policy_max_tokens_per_request'] == 1024
    monkeypatch.setattr(run, 'official_airline_task_ids', lambda: pytest.fail('opened final'))
    assert run.main(['--target', 'tau3-final', '--checkpoint', '/merged', '--dry-run',
                     '--harness-protocol', INPUTS_V3]) == 2
    assert run.main(flags + ['--policy-temperature', '0.4', '--user-temperature', '0']) == 0


@pytest.mark.parametrize('protocol', [LEGACY, CONTROL_V2, INPUTS_V3])
def test_controller_command_uses_matching_temperature(tmp_path, protocol):
    from tau3_grpo.evaluation.controller import Controller

    controller = Controller(tmp_path, tmp_path / 'out', harness_protocol=protocol)
    controller.models['sft'] = '/merged'
    controller.env['TAU3_USER_SERVED_MODEL_NAME'] = 'user'
    command = controller.evaluation_command('sft', 8200, tmp_path / 'eval', tmp_path, 4)
    assert command[command.index('--user-temperature') + 1] == '0.7'
    assert command[command.index('--policy-temperature') + 1] == '0.7'
    assert command[command.index('--harness-protocol') + 1] == protocol


@pytest.mark.parametrize('protocol', [LEGACY, CONTROL_V2, INPUTS_V3])
@pytest.mark.parametrize('failure_role', ['agent', 'user'])
@pytest.mark.parametrize('typed_context', [True, False])
def test_context_limit_preserves_native_tools_without_swallowing_bad_requests(
    monkeypatch, tmp_path, selection_record, protocol, failure_role, typed_context,
):
    from litellm.exceptions import BadRequestError, ContextWindowExceededError
    from tau2.agent import llm_agent
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.user import user_simulator

    calls = {'agent': 0, 'user': 0}
    # Legacy/v2 first ask the user after the stock assistant greeting.
    initial_user_calls = int(protocol != INPUTS_V3)

    def fail():
        error = ContextWindowExceededError if typed_context else BadRequestError
        raise error(message='context length exceeded', model='cpu-test', llm_provider='openai')

    def agent_generate(**kwargs):
        calls['agent'] += 1
        if calls['agent'] == 1:
            return AssistantMessage(role='assistant', tool_calls=[
                ToolCall(id='calc1', name='calculate', arguments={'expression': '1+1'}),
                ToolCall(id='calc2', name='calculate', arguments={'expression': '2+2'}),
            ])
        if failure_role == 'agent':
            fail()
        return AssistantMessage(role='assistant', content='Done.')

    def user_generate(**kwargs):
        calls['user'] += 1
        if calls['user'] <= initial_user_calls:
            return UserMessage(role='user', content='Please calculate.')
        fail()

    monkeypatch.setattr(llm_agent, 'generate', agent_generate)
    monkeypatch.setattr(user_simulator, 'generate', user_generate)
    summary = runtime.run_evaluation(
        spec=EvalSpec(target='selection', trials=1, max_concurrency=1, harness_protocol=protocol),
        policy=Endpoint('policy', 'http://unused'), user=Endpoint('user', 'http://unused'),
        output_dir=tmp_path, selection_entries=[selection_record],
    )
    errors = [json.loads(line) for line in (tmp_path / 'errors.jsonl').read_text().splitlines()]
    rows = [json.loads(line) for line in (tmp_path / 'trajectories.jsonl').read_text().splitlines()]
    assert calls == {'agent': 2, 'user': initial_user_calls + int(failure_role == 'user')}
    if not typed_context:
        assert not rows and len(errors) == 1
        assert errors[0]['error_type'] == 'BadRequestError'
        assert 'BadRequestError' in errors[0]['traceback']
        evidence = errors[0]['execution_evidence']
        assert evidence['task']['id'] == selection_record.task_id
        assert evidence['database'] and evidence['database_hash']
        assert evidence['to_role'] == failure_role
        assert not evidence.get('capture_errors')
        tools = [m for m in evidence['messages'] if m['role'] == 'tool']
        assert [(m['id'], m['content']) for m in tools] == [('calc1', '2.0'), ('calc2', '4.0')]
        assert not summary['metrics_valid']
        return
    assert not errors and len(rows) == 1 and summary['metrics_valid']
    row = rows[0]
    assert row['termination_reason'] == 'context_window_exceeded' and row['reward'] == 0
    assert row['execution_eligibility']['category'] == 'budget_limit'
    assert not row['execution_eligibility']['officially_scored']
    assert row['simulation']['info']['context_limit']['role'] == failure_role
    assert row['simulation']['info']['context_limit']['retry_count'] == 0
    tools = [m for m in row['simulation']['messages'] if m['role'] == 'tool']
    assert [(m['id'], m['content']) for m in tools] == [('calc1', '2.0'), ('calc2', '4.0')]


@pytest.mark.parametrize('max_steps', [2, 3, 5, 30])
def test_legacy_normal_trace_matches_native_including_budget_boundaries(
    monkeypatch, selection_record, max_steps,
):
    from tau2.agent import llm_agent
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user import user_simulator
    from tau2.utils.llm_utils import to_litellm_messages

    from tau3_grpo.envs import orchestrator as adapters

    job = next(runtime._selection_jobs([selection_record], 1, 42))
    traces = []
    for native in (False, True):
        requests, calls = [], {'agent': 0, 'user': 0}

        def capture(role, kwargs):
            assert kwargs['temperature'] == 0.7 and 'max_tokens' not in kwargs
            requests.append((role, to_litellm_messages(kwargs['messages'])))

        def agent_generate(**kwargs):
            capture('agent', kwargs)
            calls['agent'] += 1
            if calls['agent'] == 1:
                return AssistantMessage(role='assistant', tool_calls=[
                    ToolCall(id='calc', name='calculate', arguments={'expression': '1+1'}),
                ])
            return AssistantMessage(role='assistant', content='Done.')

        def user_generate(**kwargs):
            capture('user', kwargs)
            calls['user'] += 1
            content = 'Please calculate.' if calls['user'] == 1 else '###STOP###'
            return UserMessage(role='user', content=content)

        with monkeypatch.context() as patch:
            patch.setattr(llm_agent, 'generate', agent_generate)
            patch.setattr(user_simulator, 'generate', user_generate)
            if native:
                patch.setattr(adapters, 'EvaluationOrchestrator', Orchestrator)
            simulation = runtime._run_one(
                task=job['task'], db_path=job['db_path'], seed=42,
                policy=Endpoint('policy', 'http://unused'), user=Endpoint('user', 'http://unused'),
                max_steps=max_steps, max_errors=10, harness_protocol=LEGACY,
            )
        assert not simulation.info['tool_observation_receipts']
        traces.append((requests, calls, simulation.termination_reason, simulation.reward_info))
    if max_steps == 5:
        assert traces[0][:2] == traces[1][:2]  # Identical calls and messages.
        assert traces[0][2].value == 'user_stop'
        assert traces[1][2].value == 'max_steps'
    else:
        assert traces[0] == traces[1]


def test_context_limit_reason_survives_native_budget_boundary():
    from tau2.data_model.simulation import TerminationReason
    from tau2.orchestrator.orchestrator import Role

    from tau3_grpo.envs.orchestrator import EvaluationOrchestrator

    orchestrator = object.__new__(EvaluationOrchestrator)
    orchestrator.to_role = Role.AGENT
    orchestrator.done = True
    orchestrator.step_count = orchestrator.max_steps = 30
    orchestrator.num_errors = orchestrator.max_errors = 10
    orchestrator.context_limit_receipt = {'role': 'agent'}
    orchestrator.termination_reason = TerminationReason.CONTEXT_WINDOW_EXCEEDED
    orchestrator._check_termination()
    assert orchestrator.termination_reason == TerminationReason.CONTEXT_WINDOW_EXCEEDED


@pytest.mark.parametrize('reason', ['user_stop', 'agent_stop'])
@pytest.mark.parametrize('done', [True, False])
def test_only_completed_stop_bypasses_native_budget_checks(monkeypatch, reason, done):
    from tau2.data_model.simulation import TerminationReason
    from tau2.orchestrator.orchestrator import Role

    from tau3_grpo.envs.orchestrator import EvaluationOrchestrator

    o = object.__new__(EvaluationOrchestrator)
    o.to_role = Role.AGENT
    o.done = done
    o.step_count = o.max_steps = 30
    o.num_errors = o.max_errors = 10
    o.termination_reason = TerminationReason(reason)
    timeout_checks = []
    monkeypatch.setattr(o, '_check_timeout', lambda: timeout_checks.append(True))
    o._check_termination()
    if done:
        assert o.termination_reason.value == reason
        assert not timeout_checks
    else:
        assert o.termination_reason == TerminationReason.TOO_MANY_ERRORS
        assert timeout_checks == [True]
