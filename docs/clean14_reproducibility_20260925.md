# clean14 96/100：全量审计与冻结重建（2026-09-25）

## 范围与改动

实现位于 `tau3_grpo/data/build_clean14.py`，不是新的筛选实验。两份配方分别固定历史的源 dialogue ID、顺序、数量、输入哈希、tokenizer/template、prompt 协议及输出哈希：

| 配方 | 清洁 anchor | 追加 AReaL | authored supplemental | train / validation | 单 GPU 一 epoch 更新数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `configs/data/clean14_96_frozen_v1.json` | 41 | 52 | 3 | 96 / 5 | 12 |
| `configs/data/clean14_100_frozen_v1.json` | 41 | 56 | 3 | 100 / 5 | 13 |

不再改脚本常量覆盖 96/100。全部 train 和 validation 行都执行源质量、源消息身份、工具 schema/响应配对、完整对话、native tokenizer assistant-only mask 和长度检查；全部训练行还检查 heldout intent 词面重叠，train/validation ID 必须互斥且唯一。四个已知坏 anchor 排除。原始源数据缺失质量标记时拒绝，而不是默认合格。

Supplemental 使用已有执行回执，绑定 frozen TRAIN task/DB/revision、实际消息和每个工具响应哈希；不是新做一次环境执行，更不是独立 policy/task-success judge。任何一步失败不发布最终目录。输入在长时间 render 前后核验，旧数据、旧 audit 均不覆盖。

两份 SFT YAML 保留训练参数（LoRA r16、1 epoch、effective batch 8、LR 1e-4）。96 配置的默认输出从旧 `/root/shared-nvme/...` 改到持久盘 `/root/autodl-fs/tau3-core/code/results/runs/`；真正运行仍应传独立 run-id 的 `--output-dir`，本次未运行训练。

## 复现输入与命令

仓库只新增小型配方/回执，不提交新数据、模型或结果。既有 anchors、validation、manifest 不重写。完整 834MB 原始 SFT 本次读取本地 `../code_pytrio/data/raw/areal_tau2/tau2_sft_train.jsonl`，只当数据源，不使用旧副本代码。SHA256 必须等于配方值 `24bc4d479799ef5d5efec7a5388ea2de6bf5dc5ab7c2c83e9b5ee26fa5813804`；其他机器可传不同路径，但字节必须相同。

现有 blocked intents 来自 selection60 + official final50，仅用于排除泄漏，不执行 final50。可在离线环境精确重建如下（输出路径必须尚不存在）：

```bash
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
python - <<'PY'
import json
from pathlib import Path
from tau3_grpo.data.prepare_sft import _official_reasons
from tau3_grpo.data.sft import reason_texts_from_manifest
p = Path('results/analysis/clean14_rebuild_inputs/blocked_reasons.json')
p.parent.mkdir(parents=True, exist_ok=True)
rows = reason_texts_from_manifest(Path('data/manifests/areal_airline_selection_seed42.jsonl')) + _official_reasons()
with p.open('x') as f:
    f.write(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
PY
```

110 条 intent 的内容、顺序和序列化已与原输入比对一致。Supplemental 的生成配方为 `configs/data/sft_clean14_supplemental_recipes.json`；已有 CPU 生成器 `python -m tau3_grpo.data.sft_demonstrations --help` 要求 train manifest、heldout tasks、blocked reasons、DB root。源 DB、prompt 协议和执行回执均需保持原身份。此次全量重建复用已有 supplemental 文件，不声称重新执行或独立验证其政策正确性。

从仓库根执行（替换 `SOURCE`、离线 tokenizer 路径和新输出目录）：

```bash
SOURCE=../code_pytrio/data/raw/areal_tau2/tau2_sft_train.jsonl
for SIZE in 96 100; do
  python -m tau3_grpo.data.build_clean14 \
    --recipe "configs/data/clean14_${SIZE}_frozen_v1.json" \
    --source "$SOURCE" \
    --anchors data/sft/airline_sft_train_seed42.jsonl \
    --validation data/sft/airline_sft_validation_seed42.jsonl \
    --supplemental data/sft/clean14_supplemental.jsonl \
    --blocked-reasons results/analysis/clean14_rebuild_inputs/blocked_reasons.json \
    --tool-config configs/envs/tool_config.yaml \
    --train-manifest data/manifests/areal_airline_train_seed42.jsonl \
    --tokenizer models/Qwen3.5-4B \
    --output-dir "results/analysis/clean14_rebuild_${SIZE}_new"
done
```

配方锁定的是历史数据字节；不要为通过检查而更新哈希。任何新数据、template 或筛选规则必须另建有明确身份的配方。

## 本次实际证据

两次真实全量 CPU/native-tokenizer 重建均通过：分别审计 101 / 105 行，且 train JSONL 与历史 96 / 100 逐字节一致，validation 亦一致。新产物位于 `results/analysis/review_repairs_20260925/clean14_{96,100}_verified/`；精确 SHA256、builder 身份和审计文件哈希登记于 `env_info/clean14_rebuild_20260925.json`。

回归测试 `tests/test_build_clean14.py` 已登记 benchmark 层，覆盖缺失/失败源质量、anchor 与 supplemental、源内容篡改、split 重叠、schema、mask、provenance、输入哈希、配方数量、训练预算和拒绝覆盖。与 SFT data/expansion/thinking 测试一起执行。

边界：词面过滤不保证语义零泄漏；source 的 correct/reward 只覆盖已供给源行的标签，不保证全部历史 context 的每步正确；14 工具覆盖不代表能力正确、更不代表训练后提升。本地 CPU 重建不能替代远程固定环境或 GPU 训练验收。

本次测试结果：新增 clean14 + SFT data/expansion/thinking 定向测试 **27 passed**；benchmark 全层 **534 passed**（29.38s，1 条 Python audioop 弃用 warning）；Ruff 全仓基线检查无新增诊断（既存 382 条不在本次扩散）。所有测试在本地 CPU/offline 配置下运行。
