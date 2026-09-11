# Configurable Qwen3.5 thinking SFT

The default remains answer/tool/EOS supervision with all three data flags false.
`configs/train/sft/qwen35_4b_lora_thinking.yaml` reproduces the independent 2026-09-11
thinking experiment: same original 4B, 45/5 split, LoRA r16/alpha32, lr1e-4 and 30 steps.
It maps existing dataset `reasoning` into `reasoning_content`, retains historical
reasoning, and supervises reasoning as well as answers/tools/EOS. No new reasoning
is generated, no source records are mutated, and over-length examples fail.

`enable_thinking` enables Qwen3.5 reasoning input; `supervise_reasoning` labels it;
`preserve_historical_reasoning` retains it before the last user query. The latter
two require the first. Defaults preserve the previous tokens and labels exactly.
Historical reasoning retention changes training context relative to the native
inference template, which removes older reasoning. This is an explicit experiment
choice, not a claim that an inference-only switch reproduces the experiment.

The initial isolated run is `ladskre3` on SwanLab, versus answer-only `ovrld9g3`.
Training totals are 600731 input / 212295 labeled tokens, versus 485420 / 96660.
Raw losses therefore have different targets and cannot rank the models directly.
The running historical experiment must keep its source snapshot; adopting these
interfaces is for subsequent runs, not restarting or relabeling that run.

## Comparable evaluation

Run `python -m tau3_grpo.training.sft.evaluate --config <training.yaml>
--model <adapter-directory> --base-model <exact-base-directory>
--output <new-report.json> [--baseline-summary <old-train_summary.json>]`.
For a full SFT model, omit `--base-model`. This is GPU evaluation only, not training.
The evaluator always renders the original answer-only target regardless of the
training YAML's thinking flags. Reports include source/rendered/tool hashes,
token counts and loss. A legacy baseline must have matching token counts and no
thinking supervision; missing historical source hashes remain a provenance limit.
Choose a new report filename for each run. Lower answer loss is not proof of higher
task success; independently evaluate task success and inference cost separately.
The evaluator does not select new checkpoints or overwrite the training result.

Integration validation on the five-A800 node: all 50 frozen dialogues matched the
isolated thinking experiment token-for-token and label-for-label; the off branch
matched the original implementation as well. 31 focused dataset/launcher tests
passed. An additional native-tokenizer check passed for 4B; three other tokenizer
variants were unavailable on this node and skipped. The evaluator reproduced the
completed off model's loss exactly: 0.4231073260307312. These checks do not promise
bitwise-identical weights from a new training run.
