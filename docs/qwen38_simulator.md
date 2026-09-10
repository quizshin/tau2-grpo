# Qwen3.8 外部用户模拟器

此配置仅用于冻结的用户模拟器，策略训练仍使用现有 Qwen2.5 / Qwen3.5
适配。它不代表已经验证 Qwen3.8 全量训练或 LoRA 训练。

选用社区量化模型 `cyankiwi/Qwen3.8-27B-AWQ-INT4`，对应官方基础模型
`Qwen/Qwen3.8-27B`。固定 revision：
`63768c10df38c0395e12ef49edac1bd539eaeeea`。
五个权重分片合计 19.5745 GiB；实际格式为 compressed-tensors W4A16，
group size 32、非对称 INT4，部分层保持非量化。不能按 27B × 0.5 byte
估计总权重，也不能从量化名称直接推断 A800 吞吐。

下载后的 safetensors header 统计：语言 BF16 张量 7.0490 GiB、语言 INT32
打包张量 11.6670 GiB、视觉 BF16 0.8582 GiB。纯文本服务跳过视觉部分，
但仍有约 18.716 GiB 的语言权重存储；这是文件中的张量大小，不是 GPU 峰值。

模型的配置架构仍为 `Qwen3_5ForConditionalGeneration`，但官方部署说明要求
Transformers >= 5.8.0。本配置使用独立的 Transformers 5.8.0 覆盖环境，
只读复用现有 Torch 2.11.0 / vLLM 0.20.0 / compressed-tensors 0.15.0.1。
训练环境的 Transformers 5.5.1 保持不变。

## 下载与启动

以下路径用于当前 AutoDL 布局。fs 上保存代码、环境与模型的持久副本；
运行时可将依赖缓存到数据盘，并通过变量指定。2026-09-08 准备部署时 fs
发生过小文件写入 `errno 5`，因此下载与覆盖环境临时使用 `/root/tau3-staging`。
系统盘临时副本不保证跨服务器保留，不能代替 fs 上已校验的持久副本。

```bash
source /root/autodl-fs/tau3_grpo_fix/activate.sh
cd /root/autodl-fs/tau3_grpo_fix/code
bash env_info/setup_qwen38_simulator.sh
python -m tau3_grpo.models.download_simulator \
  --output /root/autodl-fs/tau3_grpo_fix/model_store/Qwen3.8-27B-AWQ-INT4
bash scripts/serve/simulator_qwen38.sh
```

下载默认从 ModelScope 镜像读取，必须匹配固定 HF revision 的 SHA-256；
支持断点续传，全部 17 个文件验证完成后才写 `tau3_source_revision.json`。
可以用 `--source hf-mirror` 或 `--source huggingface` 更换来源。

启动默认 GPU 1、TP=1、端口 8100、BF16 激活、16k 上下文、16 并发、
显存利用率 0.65、prefix caching、关闭 thinking。80 GiB 上 0.65 约为
52 GiB 的 vLLM 预算，含权重和缓存，并非模型自身需要 52 GiB。
`TAU3_USER_GPU_MEMORY_UTILIZATION`、`TAU3_USER_MAX_NUM_SEQS`、
`TAU3_USER_MAX_MODEL_LEN` 可以调整。A800 不使用原生 FP8/NVFP4 启动配置。

覆盖位置示例：

```bash
TAU3_SIM_VENV=/root/tau3-staging/qwen38/venv \
TAU3_RUNTIME_CACHE=/root/autodl-tmp/tau3/runtime/qwen35 \
UV_CACHE_DIR=/root/tau3-staging/qwen38/uv-cache \
  bash env_info/setup_qwen38_simulator.sh
TAU3_SIM_PYTHON=/root/tau3-staging/qwen38/venv/bin/python \
TAU3_USER_MODEL=/root/tau3-staging/models/Qwen3.8-27B-AWQ-INT4 \
TAU3_SIM_CACHE_ROOT=/root/tau3-staging/qwen38/cache \
  bash scripts/serve/simulator_qwen38.sh
```

## RL 接入

在启动策略训练的进程中设置：

```bash
export TAU3_USER_SERVED_MODEL_NAME=Qwen/Qwen3.8-27B-AWQ-INT4
export TAU3_USER_BASE_URL=http://127.0.0.1:8100/v1
export TAU3_USER_THINKING=off
```

交互配置显式关闭模拟器 thinking，兼容服务端 alias；不会把任意 Qwen3.8
名称当作已支持的策略模型。Qwen2.5 和 Qwen3.5 原默认行为不变。

## 验证与资源分配

部署验收需要真实非空 chat、Airline 用户模拟多轮对话、1/4/8 并发的延迟与
输出 tokens/s、GPU 峰值，不能仅检查端口。正式更换模拟器后，全量和 LoRA
对照必须使用相同模拟器版本、采样设置与交互预算。

`python -m tau3_grpo.models.benchmark_simulator --prompts prompts.json --output result.json`
可测试 OpenAI-compatible 本地端点；输入是含 `id`、`messages` 的 JSON 数组。
默认测试 1/4/8 并发、256 输出 token 上限，保留完整响应 JSONL、TTFT、
P50/P95 延迟与截断计数。重复使用输入且启用 prefix cache，结果代表热缓存
吞吐；不能直接推断冷启动或完整 RL 轨迹每小时吞吐。

### 2026-09-08 单卡 A800 实测

17 个文件全部 SHA-256 校验通过；Transformers 5.8.0 分词器和量化 schema
通过；真实 `tau2.UserSimulator` 三轮对话通过（正确返回场景中的用户 ID、
预订号、日期和支付偏好）。原 vLLM `qwen3` reasoning parser 曾把非思考正文
全放入 reasoning 字段，导致 content 为空；本模拟器已移除该 parser，仍在
chat template 中明确 `enable_thinking=false`，流式和非流式均验证正文正常。

GPU 模型加载实测 18.23 GiB，五片权重读取约 4.3 秒。显存利用率 0.65 时，
服务总占用快照 52,815 MiB（51.58 GiB），其余大部分用于推理缓存。
这是服务占用快照，不是训练时与模拟器共置的峰值保证。

16 条原生 Airline 模拟器输入，每组输出上限 256 tokens，温度 0、prefix
caching 开启，先预热再顺序测试各并发；无空正文、thinking 泄漏或输出截断。

| 并发 | 总输出 tokens/s | 请求/秒 | P50 延迟 | P95 延迟 | 平均首 token 延迟 |
|---|---:|---:|---:|---:|---:|
| 1 | 45.27 | 0.89 | 1.06 秒 | 1.69 秒 | 0.39 秒 |
| 4 | 132.42 | 2.58 | 1.17 秒 | 3.04 秒 | 0.34 秒 |
| 8 | 170.81 | 3.35 | 1.90 秒 | 4.12 秒 | 0.77 秒 |

此测量为 16 条较短输入、重复输入的热缓存测试，不代表 16k 满上下文性能，
也不证明量化模型与官方 BF16 的任务成功率等价。0.8B + 4B RL 三步验证后，
服务已恢复在 GPU 1、`http://127.0.0.1:8100/v1`；重启后真实 chat 返回非空
正文，服务 PID 为 40573（进程号只描述本次启动）。运行环境/模型仍使用
上文的 `/root/tau3-staging` 临时位置，持久 fs 副本尚未完成。

先在 GPU 0 放 0.8B 或 9B LoRA 策略、GPU 1 独立模拟器可简化调试。
4B 全量 FP32 Adam 静态状态约 64 GiB，长上下文还需激活与临时 logits，
与 27B 模拟器同机时不能沿用“策略一张卡、模拟器一张卡”就保证能跑。
应先验证两卡 FSDP、模拟器较低缓存预算和训练阶段的 rollout 释放；
若峰值不足，需要真正的 CPU offload 或外置模拟器。参见
[两卡预算](rl_model_budget_2xa800.md)。

来源：
- https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://recipes.vllm.ai/Qwen/Qwen3.8-27B
