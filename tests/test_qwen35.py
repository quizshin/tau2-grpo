"""Qwen3.5 regressions using native tokenizers and tiny real model weights.

Tokenizer tests require revision-pinned assets fetched by download_qwen35;
model tests instantiate tiny random weights, not a downloaded policy.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from tau3_grpo.models.compat import load_policy_model, lora_target_modules, model_family
from tau3_grpo.models.qwen35_template import supervised_template
from tau3_grpo.paths import CODE_ROOT, PROJECT_ROOT
from tau3_grpo.training.sft.dataset import build_supervised_example


def tokenizer(name):
    transformers = pytest.importorskip("transformers")
    path = Path(os.environ.get("TAU3_TEST_TOKENIZERS", PROJECT_ROOT / "models")) / name
    if not (path / "tokenizer.json").is_file():
        pytest.skip(f"Fetch pinned tokenizer assets with download_qwen35: {path}")
    return transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)


MESSAGES = [
    {"role": "system", "content": "SYSTEM_ONLY policy"},
    {"role": "assistant", "content": "Welcome."},
    {"role": "user", "content": "CUSTOMER_ONLY question"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_user_details",
                    "arguments": {"user_id": "0012"},
                },
            }
        ],
    },
    {"role": "tool", "content": "OBSERVATION_ONLY result"},
    {"role": "assistant", "content": "Please confirm."},
    {"role": "user", "content": "CUSTOMER_CONFIRMATION_ONLY"},
    {"role": "assistant", "content": "Done."},
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_details",
            "description": "Get customer details",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    }
]


@pytest.mark.parametrize(
    "name", ["Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen2.5-7B-Instruct"]
)
def test_native_multiturn_labels_exclude_observations(name):
    tok = tokenizer(name)
    result = build_supervised_example(MESSAGES, tok, tools=TOOLS)
    kwargs = {"enable_thinking": False} if name.startswith("Qwen3.5") else {}
    from tau3_grpo.models.qwen35_template import token_ids

    assert result["input_ids"] == token_ids(
        tok.apply_chat_template(
            MESSAGES, tools=TOOLS, tokenize=True, add_generation_prompt=False, **kwargs
        )
    )
    labeled = tok.decode([label for label in result["labels"] if label != -100])
    for excluded in (
        "SYSTEM_ONLY",
        "CUSTOMER_ONLY",
        "OBSERVATION_ONLY",
        "CUSTOMER_CONFIRMATION_ONLY",
        "<think>",
    ):
        assert excluded not in labeled
    for included in (
        "Welcome.",
        "get_user_details",
        "0012",
        "Please confirm.",
        "Done.",
        "<|im_end|>",
    ):
        assert included in labeled
    with pytest.raises(ValueError, match="exceeding max_length"):
        build_supervised_example(MESSAGES, tok, tools=TOOLS, max_length=10)


def test_unknown_template_fails_closed():
    with pytest.raises(ValueError, match="Unsupported Qwen3.5"):
        supervised_template("an unreviewed template")


def test_model_detection_uses_checkpoint_config_not_directory_name(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert model_family(tmp_path) == "qwen35"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    assert model_family(tmp_path) == "legacy"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    with pytest.raises(ValueError, match="dense"):
        model_family(tmp_path)


@pytest.mark.parametrize(
    "name,parser_name,payload",
    [
        (
            "Qwen3.5-0.8B",
            "qwen3_coder",
            "<tool_call>\n<function=get_user_details>\n<parameter=user_id>0012</parameter>\n</function>\n</tool_call>",
        ),
        (
            "Qwen2.5-7B-Instruct",
            "hermes",
            '<tool_call>{"name":"get_user_details","arguments":{"user_id":"0012"}}</tool_call>',
        ),
    ],
)
def test_actual_verl_tool_parser_preserves_string_ids(name, parser_name, payload):
    tok = tokenizer(name)
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema

    parser = ToolParser.get_tool_parser(parser_name, tok)
    _, calls = asyncio.run(
        parser.extract_tool_calls(
            tok.encode(payload, add_special_tokens=False),
            [OpenAIFunctionToolSchema.model_validate(TOOLS[0])],
        )
    )
    assert len(calls) == 1
    assert calls[0].name == "get_user_details"
    assert json.loads(calls[0].arguments) == {"user_id": "0012"}


def tiny_model(tied=False):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5Config"):
        pytest.skip("Tiny Qwen3.5 model checks require the separate Transformers 5 environment")
    config = transformers.Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "tie_word_embeddings": tied,
            "pad_token_id": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
        },
        tie_word_embeddings=tied,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    config._attn_implementation = "sdpa"
    return transformers.Qwen3_5ForConditionalGeneration(config)


@pytest.mark.parametrize("tied", [True, False])
def test_full_sft_updates_language_weights_and_exports_bf16(tmp_path, tied):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models

    from tau3_grpo.training.sft.merge import main as export_sft

    torch.manual_seed(42)
    tiny_model(tied).save_pretrained(tmp_path / "base")
    model = load_policy_model(str(tmp_path / "base"), torch_dtype=torch.float32,
                              attn_implementation="sdpa")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    optimizer.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(before[n], p)}
    assert any("embed_tokens" in n for n in changed)
    assert any(".linear_attn." in n for n in changed)
    assert any(".self_attn." in n for n in changed)
    assert not any("visual" in n for n in changed)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    assert all(state["exp_avg"].dtype == torch.float32 for state in optimizer.state.values())
    model.save_pretrained(tmp_path / "full")
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1},
                                                     unk_token="<unk>")),
        unk_token="<unk>",
    )
    tok.save_pretrained(tmp_path / "full")
    assert export_sft(["--base", str(tmp_path / "base"), "--adapter", str(tmp_path / "full"),
                       "--output", str(tmp_path / "export")]) == 0
    restored = load_policy_model(str(tmp_path / "export"), torch_dtype=torch.bfloat16)
    assert restored.config.tie_word_embeddings is tied
    assert not (tmp_path / "export" / "adapter_config.json").exists()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value.to(torch.bfloat16),
                                   rtol=0, atol=0)


@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("tied", [True, False])
def test_best_checkpoint_restores_exact_weights_after_pretrained_load(tmp_path, method, tied):
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors import safe_open
    from transformers import TrainingArguments

    from tau3_grpo.training.sft.trainer import Qwen35SFTTrainer

    tiny_model(tied).save_pretrained(tmp_path / "base")
    model = load_policy_model(str(tmp_path / "base"), torch_dtype=torch.float32)
    if method == "lora":
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
            target_modules=lora_target_modules(model, "qwen35_language_linear"),
        ))
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.01)
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    trainer = Qwen35SFTTrainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path / "run"), use_cpu=True, report_to="none",
    ))
    checkpoint = tmp_path / "best"
    trainer.save_model(str(checkpoint))
    if method == "full":
        with safe_open(checkpoint / "model.safetensors", framework="pt") as saved:
            assert set(saved.keys()) <= set(expected)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1.0)
    trainer.state.best_model_checkpoint = str(checkpoint)
    trainer.state.best_metric = 0.5
    trainer._load_best_model()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("tied", [True, False])
def test_sft_supervised_projection_matches_native_loss_and_gradients(method, tied):
    torch = pytest.importorskip("torch")
    from tau3_grpo.training.sft.trainer import Qwen35SFTTrainer

    torch.manual_seed(42)
    model = tiny_model(tied)
    if method == "lora":
        peft = pytest.importorskip("peft")
        model = peft.get_peft_model(model, peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=0,
            target_modules=lora_target_modules(model, "qwen35_language_linear"),
        ))
    reference = copy.deepcopy(model)
    inputs = {"input_ids": torch.tensor([[2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9]]),
              "labels": torch.tensor([[-100, -100, 4, -100, -100, 7],
                                       [-100, -100, -100, 7, -100, -100]])}
    expected = reference(**inputs, use_cache=False).loss
    trainer = Qwen35SFTTrainer.__new__(Qwen35SFTTrainer)
    actual, output = trainer.compute_loss(model, inputs, return_outputs=True)
    assert output.logits.shape[1] == 3
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    expected.backward()
    actual.backward()
    for (_, parameter), (_, baseline) in zip(model.named_parameters(), reference.named_parameters()):
        if baseline.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, baseline.grad, rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError, match="no supervised"):
        trainer.compute_loss(model, {**inputs, "labels": torch.full_like(inputs["labels"], -100)})


@pytest.mark.parametrize("method", ["full", "lora"])
@pytest.mark.parametrize("sample_count", [2, 3])
def test_sft_accumulation_matches_direct_batch_including_partial_window(
    tmp_path, method, sample_count
):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tau3_grpo.training.sft.trainer import Qwen35SFTTrainer

    torch.manual_seed(42)
    base = tiny_model()
    base.config.use_cache = False
    if method == "lora":
        peft = pytest.importorskip("peft")
        base = peft.get_peft_model(base, peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=0,
            target_modules=lora_target_modules(base, "qwen35_language_linear"),
        ))
    samples = [{"input_ids": list(range(2 + i, 8 + i)),
                "labels": list(range(2 + i, 8 + i))} for i in range(sample_count)]
    results = []
    for batch, accumulation in [(2, 1), (1, 2)]:
        model = copy.deepcopy(base)
        args = transformers.TrainingArguments(
            output_dir=str(tmp_path / f"batch{batch}"), use_cpu=True,
            per_device_train_batch_size=batch, gradient_accumulation_steps=accumulation,
            max_steps=(sample_count + 1) // 2, learning_rate=0.1,
            lr_scheduler_type="constant", optim="sgd", max_grad_norm=0,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            save_strategy="no", report_to="none", disable_tqdm=True, seed=42, data_seed=42,
        )
        trainer = Qwen35SFTTrainer(model=model, args=args, train_dataset=samples,
                                  data_collator=transformers.default_data_collator)
        trainer.train()
        assert trainer.state.global_step == (sample_count + 1) // 2
        results.append({n: p.detach().clone() for n, p in model.named_parameters()})
    assert any(not torch.equal(results[0][n], p) for n, p in base.named_parameters())
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("tied", [True, False])
def test_tiny_qwen35_backward_lora_merge_and_reload(tmp_path, tied):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    torch.manual_seed(42)
    model = tiny_model(tied)
    model.save_pretrained(tmp_path / "base")
    model = load_policy_model(str(tmp_path / "base"), attn_implementation="sdpa")
    assert not any(p.requires_grad for p in model.model.visual.parameters())
    targets = lora_target_modules(model, "qwen35_language_linear")
    assert any(".linear_attn." in n for n in targets)
    assert any(".self_attn." in n for n in targets)
    assert not any("visual" in n or "lm_head" in n for n in targets)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    # Full-parameter RL-style backward through both attention types.
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if ".linear_attn." in n
    )
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    adapted = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            lora_dropout=0,
            target_modules=targets,
        ),
    )
    optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=1e-3)
    adapted(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    optimizer.step()
    adapted.eval()
    with torch.no_grad():
        expected = adapted(input_ids=ids, use_cache=False).logits
    merged = adapted.merge_and_unload()
    merged.save_pretrained(tmp_path / "merged")
    restored = load_policy_model(str(tmp_path / "merged"), attn_implementation="sdpa").eval()
    with torch.no_grad():
        actual = restored(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert restored.config.tie_word_embeddings is tied


def test_qwen35_refuses_generic_packing():
    model = tiny_model()
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    original = model.__class__.forward
    apply_monkey_patch(model, use_remove_padding=False)
    assert model.__class__.forward is original
    with pytest.raises(ValueError, match="padded native forward"):
        apply_monkey_patch(model, use_remove_padding=True)


def test_native_gated_attention_keeps_padded_batch_rows_independent():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    model = tiny_model().eval()
    ids = torch.tensor([[0, 0, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    mask = (ids != 0).long()
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        batch = model(
            input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False
        ).logits
        for row in range(2):
            tokens = ids[row, mask[row].bool()].unsqueeze(0)
            single = model(input_ids=tokens, use_cache=False).logits
            torch.testing.assert_close(
                batch[row, -tokens.shape[1] :], single[0], rtol=1e-4, atol=1e-5
            )


def test_verl_logprob_forward_accepts_native_qwen35():
    torch = pytest.importorskip("torch")

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    model = tiny_model().eval()
    # Exercise the real actor forward without constructing a distributed GPU worker.
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.use_prefix_grouper = False
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.device_name = "cpu"
    actor.param_dtype = torch.bfloat16
    actor.config = type("Config", (dict,), {"__getattr__": dict.__getitem__})(
        entropy_checkpointing=False, calculate_sum_pi_squared=False
    )
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    data = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(5).unsqueeze(0),
        "responses": ids[:, -2:],
    }
    output = actor._forward_micro_batch(data, temperature=1.0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(
            input_ids=ids,
            attention_mask=data["attention_mask"],
            position_ids=data["position_ids"],
            use_cache=False,
        ).logits[:, -3:-1]
    expected = (
        logits.float().log_softmax(-1).gather(-1, data["responses"].unsqueeze(-1)).squeeze(-1)
    )
    torch.testing.assert_close(output["log_probs"].float(), expected, rtol=1e-2, atol=2e-2)
    (-output["log_probs"].mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_legacy_qwen25_model_loader_still_uses_causal_lm(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    transformers.Qwen2ForCausalLM(config).save_pretrained(tmp_path)
    loaded = load_policy_model(str(tmp_path), attn_implementation="sdpa")
    assert isinstance(loaded, transformers.Qwen2ForCausalLM)
    ids = torch.tensor([[2, 3, 4, 5]])
    loaded(input_ids=ids, labels=ids).loss.backward()
    assert loaded.model.layers[0].self_attn.q_proj.weight.grad is not None


@pytest.mark.parametrize("size", ["0.8B", "9B"])
def test_launcher_dry_run_preserves_grouping_and_writes_nothing(tmp_path, size):
    env = dict(
        os.environ, TAU3_DRY_RUN="1", QWEN35_RUN_ROOT=str(tmp_path / "absent"), QWEN35_SIZE=size
    )
    proc = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/train/rl/run_qwen35.sh"), "e3", "43"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        cwd=CODE_ROOT,
    )
    args = shlex.split(proc.stdout.strip().splitlines()[-1])
    values = {arg.split("=", 1)[0].lstrip("+"): arg.split("=", 1)[1] for arg in args if "=" in arg}
    assert values["algorithm.adv_estimator"] == "tau_gigpo"
    assert values["actor_rollout_ref.rollout.n"] == "4"
    assert values["data.train_batch_size"] == "2"
    assert values["actor_rollout_ref.rollout.multi_turn.format"] == "qwen3_coder"
    assert values["actor_rollout_ref.rollout.load_format"] == "safetensors"
    assert values["actor_rollout_ref.model.use_remove_padding"] == "false"
    assert values["trainer.save_freq"] == "1"
    assert values["trainer.n_gpus_per_node"] == "1"
    assert not (tmp_path / "absent").exists()
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(CODE_ROOT / "verl/verl/trainer/config"), version_base=None
    ):
        cfg = compose(config_name="ppo_trainer", overrides=args[3:])
    assert cfg.data.apply_chat_template_kwargs.enable_thinking is False
    assert cfg.actor_rollout_ref.actor.freeze_vision_tower is True
    # The uncast FP32 embedding must fit in one real weight-transfer bucket.
    model_config_path = PROJECT_ROOT / "models" / f"Qwen3.5-{size}" / "config.json"
    if model_config_path.is_file():
        config = json.loads(model_config_path.read_text())["text_config"]
        embedding_bytes = config["vocab_size"] * config["hidden_size"] * 4
        bucket_bytes = (
            cfg.actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes << 20
        )
        assert bucket_bytes >= embedding_bytes


def test_qwen35_uses_tokenizer_only_when_explicitly_requested(monkeypatch):
    tok = tokenizer("Qwen3.5-0.8B")
    from verl.utils.tokenizer import hf_processor

    monkeypatch.setenv("TAU3_GRPO_TEXT_ONLY", "1")
    assert hf_processor(tok.name_or_path, local_files_only=True) is None


def test_qwen35_tool_observation_renders_without_a_user_turn():
    tok = tokenizer("Qwen3.5-0.8B")
    from verl.utils.chat_template import apply_chat_template
    from verl.utils.tokenizer import normalize_token_ids

    output = apply_chat_template(
        tok, [{"role": "tool", "content": "RESULT_ONLY"}], tools=TOOLS, enable_thinking=False
    )
    rendered = tok.decode(normalize_token_ids(output))
    assert "RESULT_ONLY" in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_eval_wrapper_uses_the_existing_final_guard_cli(tmp_path):
    # Capture arguments before any evaluation/data loading. The real parser
    # checks that the wrapper preserves the final target and result namespace.
    import sys

    from tau3_grpo.evaluation.run import build_parser

    binary = tmp_path / "bin"
    binary.mkdir()
    script = binary / "python"
    script.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    script.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(binary) + os.pathsep + os.environ["PATH"],
        QWEN35_RUN_ROOT=str(tmp_path / "results"),
    )
    proc = subprocess.run(
        [
            "bash",
            str(PROJECT_ROOT / "scripts/eval/run_qwen35.sh"),
            "tau3-final",
            str(tmp_path / "merged"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    command = json.loads(proc.stdout)
    assert command[:2] == ["-m", "tau3_grpo.evaluation.run"]
    args = build_parser().parse_args(command[2:])
    assert args.target == "tau3-final"
    assert args.policy_model == "Qwen/Qwen3.5-0.8B"
    assert args.user_model == "Qwen/Qwen3.5-4B"
    assert args.results_dir == tmp_path / "results"
