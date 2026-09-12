"""Resolve the performance candidate and validate Hydra; never launch RL."""
import json
import os
from pathlib import Path
from string import Template
import subprocess

from tau3_grpo.launch import load_config, prepare
from tau3_grpo.paths import CODE_ROOT

profile = CODE_ROOT/'configs/train/rl/qwen35_4b_full_a800_perf_candidate.yaml'
env = os.environ.copy()
for key, value in load_config(profile)['launch']['environment'].items():
    env[key] = Template(str(value)).substitute(env)
command, env, snapshot = prepare('rl', profile, 'e0', 42, [], env)
assert int(env['TOTAL_UPDATES'])*int(env['GROUPS_PER_UPDATE'])*int(env['GROUP_SIZE']) == 5120
assert env['MAX_NUM_SEQS'] == '16'
assert env['MODEL_PATH'].endswith('/sft-merged/new-off')
env['TAU3_DRY_RUN'] = '1'
shell_command = subprocess.check_output(command, env=env, cwd=CODE_ROOT, text=True)
assert 'tool_execution_mode=sequential' in shell_command
assert 'enable_thinking=false' in shell_command
out = Path(os.environ['TAU3_RUN_ROOT'])/'performance-20260912'
out.mkdir(parents=True, exist_ok=True)
(out/'candidate.launch.json').write_text(json.dumps(snapshot, indent=2))
(out/'candidate.command.txt').write_text(shell_command)
# --cfg job resolves all Hydra override keys but exits before main() trains.
env.pop('TAU3_DRY_RUN')
resolved = subprocess.check_output(command+['--cfg', 'job'],env=env,cwd=CODE_ROOT,text=True)
(out/'candidate.hydra.yaml').write_text(resolved)
print('Candidate resolved; 5,120 trajectories; no service or RL job launched.')
