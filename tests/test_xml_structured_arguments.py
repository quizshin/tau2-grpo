import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex


@pytest.fixture
def parse():
    # Load the real method without importing the GPU/Ray runtime on CPU hosts.
    path = Path(__file__).parents[1] / 'verl/verl/experimental/agent_loop/tool_parser.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3XMLToolParser')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_parse_xml_function_call')
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    env = dict(ast=ast, json=json, logger=logging.getLogger(__name__),
               Optional=list, OpenAIFunctionToolSchema=object, Any=object, FunctionCall=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), env)
    obj = SimpleNamespace(tool_call_parameter_regex=regex.compile(r'<parameter=(.*?)</parameter>|<parameter=(.*?)$', regex.DOTALL))
    schema = SimpleNamespace(type='function', function=SimpleNamespace(name='test', parameters=SimpleNamespace(
        properties={'value': SimpleNamespace(model_dump=lambda: {'type': 'array'})})))
    return lambda value: json.loads(env['_parse_xml_function_call'](obj, f'test><parameter=value>{value}</parameter>', [schema]).arguments)['value']


@pytest.mark.parametrize('value,expected', [('[1, true, null]', [1, True, None]), ("['a', 'b']", ['a', 'b']),
    ("{'a', 'b'}", "{'a', 'b'}"), ("[{'a', 'b'}]", "[{'a', 'b'}]"), ('(1, 2)', '(1, 2)')])
def test_structured_arguments_remain_serializable_without_inventing_values(parse, value, expected):
    assert parse(value) == expected


def test_model_output_cannot_execute_python(parse, tmp_path):
    marker = tmp_path / 'executed'
    source = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    assert parse(source) == source
    assert not marker.exists()
