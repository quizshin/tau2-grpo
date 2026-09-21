# 调用 token 归属：本地 CPU 实现与实验（2026-09-21）

第一步已接入本地原生 agent loop，属于可选观测功能；尚未修改 MT-GTPO 公式、奖励分类或训练 loss，未运行 GPU 模型、采样或训练，未同步远程生产环境。

## 实现

`TAU3_RECORD_CALL_ATTRIBUTION=1` 启用，默认关闭。该选项同时开启轨迹事实记录；agent manager 在跨 worker 分组前保留真实 group/trial 身份。以后通过 Ray 启动时，驱动进程和 worker runtime env 都须收到该选项，当前正式训练配置未自动启用。

- `tau3_grpo/data/call_attribution.py` 扫描实际生成 IDs，定位工具块；标签 ID 读取实际 tokenizer，不硬编码。独立 schema 为 `tau3_call_attribution_v1`，坐标为 response 数组半开区间。
- Qwen 原生 parser 新增可选 provenance 接口，使用同次解析的函数顺序和原始块来源。原有双返回值接口保留。每个块的原始 token 切片经 decode 与解析器来源文本核对，不重新 encode 冒充历史 token。
- loop 将来源与既有稳定 call ID 绑定，记录开始/执行状态、错误及执行后 DB hash；顺序执行时额外只读取得执行前 hash。before 快照不可用只记录原因，不改变调用执行。
- 调用归属存入独立 `AgentData.call_attributions`，仅发布到 `trajectory_facts_json.turns[].call_attribution`，不污染 `process_reward_json` 的原始轮记录。
- 保留发出 IDs、发出/保留区间与丢弃数量。工具块与其他生成区间覆盖该轮；其他区间包括说明文字、空白和 EOS，不统一称为推理文本。
- `exact / incomplete / ambiguous / unavailable` 描述边界可靠性，与 `not_executed / started / executed` 分开。已执行报错仍可有精确边界；精确边界不意味着业务正确。
- 一个块多个函数、嵌套、孤立闭合标签等不强行分配调用级区间。未闭合但旧 parser 接受的调用保持原执行行为，边界记为 incomplete；部分裁剪或未执行调用不标为可用于调用级信用。
- 空块不会吞掉后面的有效调用；空块本身仍保留，不虚构 call ID。前面的调用执行后，尾部未执行调用不会继承其资格。

`eligible_for_call_credit` 只表示记录具备精确、保留完整、绑定执行 ID 的归属证据，不表示实现了调用级训练，也不认定错误调用应该得到某个优势符号。

## 本地 CPU 验证

使用既有 `.venv-cpu`、`CUDA_VISIBLE_DEVICES=''` 和离线依赖，不开启模型服务。

1. 真正的 Qwen tokenizer 构造测试输入，覆盖中文、多调用、重复调用、EOS、未闭合 XML、解析异常、多个函数、裁剪、非法 mask/token 身份，以及同一 parser 上并发请求隔离。
2. 原生航空环境对照：同一轮两次取消中夹一个未知工具报错；GRPO、GiGPO、MT 三条路径开关观测前后，调用顺序、私有 DB、官方奖励、真实 IDs、mask 与事实记录一致，MT 的过程奖励、return 和优势相同。新增归属可以分别定位三个调用及状态变化。
3. 完整原生 `loop.run` 发布对照：三条算法路径的输入/输出 IDs、mask、logprob、终局分数一致；三次真实 calculator 执行包含除零报错，新增归属准确连接执行 ID。此处策略和用户回复是确定性脚本，环境、工具和验证器为真实代码，不是模型表现评测。
4. 原 parser 快照对照实验：200 种构造输出，新旧普通接口和新 provenance 接口的文本/调用结果完全一致；包含 132 个解析调用。60 个调用精确归属、40 个未闭合、16 个来源不可用、16 个同块多函数归属不唯一。598 个不同保留长度检查通过，覆盖分区和裁剪身份。

实验样本故意包含大量畸形格式，以上比例不是线上覆盖率。原始快照在 `preimage/`，不覆盖既有代码变更；未重新分词旧训练轨迹以伪造调用级 mask。

新增/扩展测试登记入 CPU suites；vendor 清单使用固定上游 revision 的原始 tar 包重建，补丁原因和测试已登记。最终相关回归 **214 passed，0 failed，0 skipped**，耗时 158.92 秒。修改文件静态检查通过，vendor 核验无漂移。全仓 lint 仍有其他工作区文件的 7 条诊断（sft_expansion.py、training/rl/runner.py、test_simulator_sleep.py），本次未改这些文件，也未宣称全仓 lint clean。详见交付回执。

## 复现与产物

产物目录 `results/analysis/call_attribution_cpu_20260921/`：

- `preimage.json`、`preimage/`：本轮修改前的文件身份。
- `experiment.py`、`experiment-spec.json`、`experiment-report.json`、`experiment-cases.json`：构造输入、固定对照与逐例分类。
- `final-tests.log/xml`：相关回归测试。
- `vendor.log`、`changed-lint.log`、`lint.log`：依赖补丁核验与静态检查。
- `delivery.json`：最终结论、源码身份和验证范围。

pytest 使用仓库 `.venv-cpu` 激活后的 PATH，避免调用启动脚本时系统找不到 `python`。相关入口为 `tests/test_call_attribution.py`、`tests/test_rollout_boundaries.py`、`tests/test_token_harness.py`。

后续才是设计调用局部信用与长程信用的组合，并另做离线比较。此次交付仅解决记录归属，未证明任务成功率提升。GPU 在线覆盖率和真正模型输出仍需另行实测。
