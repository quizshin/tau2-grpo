# Harness 单卡 GPU 验证

日期：2026-09-24。用户明确要求“先在那张单卡上跑测试”。状态：单卡生成、取消、原生循环和真实采样 batch 的 halt helper 验证通过；不等于分布式训练或模型效果验收。

## 环境、预算与隔离

- 硬件：1×RTX 5090，32GB，GPU UUID `GPU-6bf60602-451d-8c13-741c-957ae1d622be`；启动前无 compute 进程。
- 环境：现有 `/root/shared-nvme/tau3/envs/qwen35`，vLLM 0.20.0、PyTorch 2.11.0+cu130、Transformers 5.5.1。
- 模型：现有 `/root/shared-nvme/tau3/models/Qwen3.5-4B`；没有下载模型，没有加载旧实验的 step 30 续训状态。
- 执行：TP1、BF16、eager、24,576 上下文、最多 4 个并发序列、显存比例 0.65；本次观察到约 21,480MiB 占用，不作为连续采样的峰值测量。
- 预算：单卡、20 分钟硬截止、最多 32 个生成请求。三阶段共 32 请求、9,999 个实际生成 token；进程运行时间合计 617.21 秒，从首次启动到最终结束 915.67 秒（含中间 CPU 核对和空闲），均在原预算内。
- 远端测试目录：`/root/shared-nvme/tau3/runs/harness_guard_gpu/20260924-single5090-01`。通过独立源码快照运行，活动 `code`、数据、模型和旧实验结果未覆盖；只清理自有进程组。

最终 `nvidia-smi`：无 compute 进程，显存 1MiB、利用率 0%。没有关闭实例，没有启动正式训练。

## 此次额外发现并修复的问题

预检已安装 vLLM 的源码发现：旧 `vLLMHttpServer.abort_request` 直接用外部 request ID 查询内部 `request_states`，手动写 abort queue，再调用私有 `abort_requests`。vLLM 0.20.0 使用外部到内部 ID 映射，私有方法还要求 `internal` 参数，因此该旧路径不可靠；先前纯 CPU 模拟流测试没有覆盖引擎接口变化。

已改为 `await self.engine.abort(request_id)`，由引擎处理 ID 映射和最终 abort 回执。guard 的单请求取消仍明确使用 `reset_prefix_cache=False`。调用成功只是取消请求被接受；只有原始 stream 的终止回执才能证明最终是 abort，不能凭 ACK 伪造模型结束。

新增 CPU 回归执行生产方法的 AST，验证只调用公开 API、不中断共享缓存、错误不伪装成成功。本轮针对取消、rollout 边界、采样身份及补丁契约的 CPU 回归 **120 passed**；vendor inventory 回归 **2 passed**，固定上游清单重建 `errors=[]`；lint 无新增诊断（既有 382），`git diff --check` 通过。

## GPU 验证结果

正式测试脚本：`tests/gpu_harness_guard_probe.py`。它连接真实 AsyncLLM，调用生产 `vLLMHttpServer.generate/abort_request`；为缩小单卡测试范围，绕过 Ray 服务分配，不绕过生成和取消实现。

重复样例使用 `allowed_token_ids` 约束为一个普通 token，确保实际 GPU 生成重复并稳定触发检测；这是工程夹具，不是模型自然重复率测量。普通短答和长文不施加该单 token 约束；长文设最小/最大 512 token，以保证取消时另一请求仍在运行。

|场景|实测结果|
|---|---|
|普通短答，guard off / abort|均自然 stop，3 token，串行 token 一致|
|非默认 repetition penalty|实际 SamplingParams 收到 1.15，正常生成|
|重复，guard off|生成到 1024，原始 finish=length|
|重复，observe|128 token 检出，仍生成到 1024，未取消|
|重复，abort|128 token 检出并收到原始 abort，保留 128 个实际 token/logprob|
|恰在检测边界自然触顶|max_tokens=128 时保留原始 length，不误记为取消|
|重复与正常长文并发|只取消重复请求；另一请求继续到 512，未触发其 guard|
|取消后再次请求|新短答正常 stop；结束时活动请求数为 0|
|共享 prefix cache|记录到的 reset 调用为 0|
|GRPO / GiGPO / MT-GTPO 原生循环|各自保留真实 128-token 前缀，终止 detail=repetition_detected；parser 未执行，零结果候选保留，会话释放|
|更新前 halt helper|真实采样构造的 GPU DataProto 触发保护，保存 pending batch 与 fault 回执，未执行优化器|

原生循环用固定 `airline_740` 任务输入验证协议边界，未调用用户模型，没有把此结果当作 selection 成绩。三种 estimator 的原生循环接入经过检查，但未运行这三个算法的 optimizer 更新。

两轮原生循环的 6 个前缀均与引擎回执逐 token、逐 logprob 相同；故障 batch 再在 CPU 加载，token 和 mask 完全一致，logprob 最大差异为 0。未添加假 EOS，未重 tokenize，未丢弃失败候选。

## 并发文本差异的追加对照

初轮正常长文并发完成，但与串行输出不逐 token 一致，因此又做了 9 请求的独立对照。对比索引从 0 起：

|比较|首次不同的 token 索引|
|---|---:|
|串行 A / 串行 B|无差异|
|串行 A / 无取消的并发正常请求|31|
|无取消并发 / 有取消并发|154|
|两次有取消的并发正常请求|无差异|
|串行 A / 全部取消测试后的串行请求|无差异|

取消发生时，正常请求已经输出 126 token。没有执行取消的并发对照也在索引 31 产生差异，证明初始串行/并发差异不以取消为必要条件；取消与不取消两种并发条件又在索引 154 分歧。批量数值或调度差异是可能解释，但本次没有单独证明底层原因。

因此，本轮通过的是“取消目标正确、其他请求继续完成、原始 token/logprob 不被改写、缓存不被全局清空”的检查。**逐 token 输出不变性没有成立，不能宣称取消完全不影响其他请求最终文本。** 后续严格复现实验须记录并发与调度条件。

## logprob 模式修正与分阶段证据

|阶段|请求|进程耗时|实际引擎 logprob 模式|用途|
|---|---:|---:|---|---|
|attempt1|14|332.58s|raw_logprobs|完整生成/取消/三算法原生循环及故障快照|
|attempt2|9|148.21s|raw_logprobs|串行与并发、不取消与取消的对照|
|attempt3|9|136.42s|processed_logprobs|与正式配置对齐后的最终复验|

前两轮测试脚本手工组装 engine 与 server，遗漏了引擎模式参数：vLLM 默认 raw，而 server 的 RolloutConfig 默认 processed，导致适配器回执中的模式标签与实际引擎不符。这是测试夹具问题；生产 `launch_server` 已显式传递 `logprobs_mode`。前两轮的实际数组完整性和终止回执仍可检查，但不能把其模式标签作为正确认证。

最终脚本显式传入 `logprobs_mode=processed_logprobs`，并通过“仅允许一个 token 时 processed logprob 为 0”的断言确认模式。attempt3 再次通过重复 observe/abort、自然 length、并发、1.15 penalty、三算法原生循环及 GPU DataProto halt；相同检测和中止仍在 128 token。旧回执原样保留，修正说明独立保存，不回写旧记录。

## 产物和后续边界

本地证据目录：`results/analysis/harness_guard_gpu_20260924/`。

- `source-attempt1.tar.gz` / `source-attempt1-sha256.json`：最初执行快照；后两轮只修改测试脚本，保存为 `gpu-probe-attempt2.py` / `gpu-probe-attempt3.py`。
- `remote-final/attempt{1,2,3}/report.json`、`engine-receipts.json`：原始断言结果、实际 ID/logprob 和取消时间。
- `remote-final/attempt{1,3}/*/facts.json`、`harness-faults/`：原生事实、实际 pending GPU batch、明确非 checkpoint 的故障回执。
- `metadata-correction.json`：前两轮模式标签的修正说明。
- `remote-final/cleanup-observation.json`、`budget.json`、各阶段 process/completion 回执：预算及释放依据。
- `final-verification.json`：49 份远端产物哈希验证、32 请求/9,999 token、6 个前缀与 batch 的一致性复核。

模型身份记录包含配置、tokenizer、索引与来源元数据 SHA256，以及权重 shard 大小/mtime；本次没有重新计算全部权重文件哈希。日志保留了引擎 fallback 和退出时未显式 destroy_process_group 的警告，最终无存活 GPU 进程的核对独立保留，不能仅凭 shutdown 日志推断释放。

尚未覆盖：Ray 服务分配和完整训练入口的真实 GPU 联调、多卡/FSDP collective 中止、optimizer 更新及恢复、CUDAGraph 执行模式、独立 HTTP 流式取消、模型成功率和长程训练稳定性。本次更新的是单卡工程验证状态，不把候选配置标为正式训练效果已验证。

## 2026-09-25 提交前本地复核

防护、完整 token harness、评测 CLI/比较、遥测、vendor patch/inventory、rollout 边界与采样身份共 213 项 CPU 定向测试通过；vendor inventory `errors=[]`。此次仅本地 CPU 复核，没有新增 GPU 实验。完整本地 veRL 套件此前存在 Qwen3.5 untied LoRA/FSDP 回归失败，本地依赖与固定远端环境不同；不把定向通过扩大为完整分布式训练验收。独立 HTTP 流式取消、奖励门控和分布式恢复仍不在本次 guard 实现范围。
