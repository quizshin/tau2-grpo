"""Lossless tool prompts through the actual veRL registration/serialization path."""

from copy import deepcopy

import cloudpickle
import pytest
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.utils.tool_registry import initialize_tools_from_config

from tau3_grpo.envs.adapter import airline_tool_schemas
from tau3_grpo.envs.generate_tool_config import FULL_SCHEMA, build_config, write_config
from tau3_grpo.envs.tools import Tau3AirlineTool
from tau3_grpo.launch import prepare
from tau3_grpo.paths import CODE_ROOT


def test_full_schema_survives_native_registry_and_worker_serialization(tmp_path):
    path = write_config(tmp_path / 'full.yaml', FULL_SCHEMA)
    tools = initialize_tools_from_config(path)
    observed = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tools]
    assert observed == airline_tool_schemas()
    booking = next(tool for tool in tools if tool.name == 'book_reservation')
    schema = booking.tool_schema
    raw = schema.model_dump()
    assert raw['function']['parameters']['$defs']['Payment']['required'] == ['payment_id', 'amount']
    assert 'items' in raw['function']['parameters']['properties']['passengers']
    assert schema.function.parameters.properties['passengers'].type == 'array'
    assert cloudpickle.loads(cloudpickle.dumps(schema)).model_dump() == raw
    raw['function']['parameters']['properties'].clear()
    assert schema.model_dump() == booking.config['schema_payload']


def test_legacy_registration_is_unchanged_and_exposes_original_loss(tmp_path):
    config = build_config()
    assert all(item['config'] == {'type': 'native'} for item in config['tools'])
    tools = initialize_tools_from_config(write_config(tmp_path / 'legacy.yaml'))
    old = [OpenAIFunctionToolSchema.model_validate(item['tool_schema']).model_dump(
        exclude_unset=True, exclude_none=True) for item in config['tools']]
    assert [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tools] == old
    booking = next(item for item in old if item['function']['name'] == 'book_reservation')
    assert '$defs' not in booking['function']['parameters']
    assert 'items' not in booking['function']['parameters']['properties']['passengers']


def test_full_payload_must_match_registered_tool_and_unknown_versions_reject():
    item = build_config(FULL_SCHEMA)['tools'][0]
    registered = OpenAIFunctionToolSchema.model_validate(item['tool_schema'])
    corrupted = deepcopy(item['config'])
    corrupted['schema_payload']['function']['name'] = 'different_tool'
    with pytest.raises(ValueError, match='differs'):
        Tau3AirlineTool(corrupted, registered)
    with pytest.raises(ValueError, match='Unknown'):
        Tau3AirlineTool({'schema_projection': 'typo'}, registered)
    with pytest.raises(ValueError, match='Unknown'):
        build_config('typo')


@pytest.mark.parametrize('arm', ['e0', 'e1', 'e2', 'e3', 'mt_gtpo'])
def test_full_schema_candidate_reaches_shared_launcher_without_changing_historical_profile(arm):
    command, env, _ = prepare('rl', CODE_ROOT / 'configs/train/rl/formal50_full_schema_v2.yaml', arm, 42, [], {'TAU3_ROOT': str(CODE_ROOT.parent)})
    assert env['TOOL_SCHEMA_VERSION'] == FULL_SCHEMA
    historical, old, _ = prepare('rl', CODE_ROOT / 'configs/train/rl/formal50_a800.yaml', arm, 42, [], {'TAU3_ROOT': str(CODE_ROOT.parent)})
    assert 'TOOL_SCHEMA_VERSION' not in old
    key = 'actor_rollout_ref.rollout.val_kwargs.temperature='
    assert [arg for arg in command if arg.startswith(key)][-1] == key + '0.7'
    assert [arg for arg in historical if arg.startswith(key)][-1] == key + '0.7'
