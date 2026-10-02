# 模型资产准备：4B、9B 策略和共享模拟器

在仓库根目录执行下面的命令。模型从原模型仓库下载，GitHub Release 提供的是
[train500/dev150 数据包](data.md)。先按[双 A800 安装](setup-a800.md)准备 Python 环境，
激活训练环境后执行下载；这些下载命令不启动模型服务或 GPU 训练。

## 固定来源与目录

|用途|模型来源（固定版本）|revision|默认本地目录|
|---|---|---|---|
|4B 策略基座|[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)|`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`|`models/Qwen3.5-4B`|
|9B 策略基座|[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)|`c202236235762e1c871ad0ccb60c8ee5ba337b9a`|`models/Qwen3.5-9B`|
|共享用户模拟器|[cyankiwi/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4/tree/63768c10df38c0395e12ef49edac1bd539eaeeea)|`63768c10df38c0395e12ef49edac1bd539eaeeea`|`models/Qwen3.8-27B-AWQ-INT4`|

策略下载器从 `configs/models/qwen35.json` 读取 revision；模拟器下载器从
`configs/simulator/qwen38_download.json` 读取 revision、文件大小和 SHA-256。
不使用浮动的 `main` 作为实验模型身份。两个策略都使用各自官方完整权重，用户模拟器使用上述社区量化权重。
源仓库总文件量约为 4B 9.34GB、9B 19.3GB；模拟器五片权重约19.57GiB。
另预留下载缓存、SFT/RL checkpoint 与 merged model 的空间；文件大小不等于 GPU 显存需求。
各模型许可和来源说明见对应模型卡；项目 MIT 许可证不替代模型原许可。

## 下载两套完整策略资产

统一指定绝对存储根目录，两套资产分开放置。以下例子默认放在当前仓库的 `models/`，
也可将第一行换成其他绝对路径。训练配置和下载命令必须使用同一个 `TAU3_MODEL_ROOT`。

```bash
export TAU3_MODEL_ROOT="$PWD/models"
python -m tau3_grpo.models.download_qwen35 --size 4B --output-root "$TAU3_MODEL_ROOT"
python -m tau3_grpo.models.download_qwen35 --size 9B --output-root "$TAU3_MODEL_ROOT"
```

完整下载包含模型 config、tokenizer、chat template、权重索引及全部 safetensors 分片。
命令结束后各目录生成 `tau3_source_revision.json`；完整下载的 `tokenizer_only` 为 false。
可重跑同一命令复用 Hugging Face 下载缓存；不要在已有实验使用的模型目录更换 revision。

如果只做离线 tokenizer 检查，用下面的命令代替完整下载：

```bash
python -m tau3_grpo.models.download_qwen35 --size 4B --tokenizer-only --output-root "$TAU3_MODEL_ROOT"
python -m tau3_grpo.models.download_qwen35 --size 9B --tokenizer-only --output-root "$TAU3_MODEL_ROOT"
```

只下载 tokenizer 无法训练/推理。后续完整下载不加 `--tokenizer-only` 即可补齐权重。
已有完整模型上再运行 tokenizer-only 会把来源回执标成 tokenizer-only，故不要用此命令
判断已有权重是否完整，应按下面的权重索引核对。

Hugging Face 不可达时，可在下载命令前显式设置 `HF_ENDPOINT=https://hf-mirror.com`；
revision 仍保持上述固定值。镜像可用性由其运营方决定，失败时切回原来源，不能改用其他 revision。
下载库和固定训练依赖由安装流程提供，不为下载随意升级训练环境。

## 核对策略文件与冻结 tokenizer

下面的 CPU 检查核对回执、模型架构、权重索引和分片是否存在，并检查冻结数据实际使用的
五个 tokenizer 文件哈希。它不加载权重，也不证明模型加载、GPU 更新或续训成功。

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['TAU3_MODEL_ROOT'])
pins = json.loads(Path('configs/models/qwen35.json').read_text())
# 数据包内 manifest 记录实际 tokenizer 文件身份。
package = Path(os.environ.get('TAU3_DATA_ROOT', 'data')) / 'sft/curriculum_500_dev150_portable_codex_20261001'
expected = json.loads((package / 'manifest.json').read_text())['tokenizer_files']
for size in ('4B', '9B'):
    model = f'Qwen/Qwen3.5-{size}'
    directory = root / f'Qwen3.5-{size}'
    receipt = json.loads((directory / 'tau3_source_revision.json').read_text())
    assert receipt['model'] == model and receipt['revision'] == pins[model], model
    config = json.loads((directory / 'config.json').read_text())
    assert config['model_type'] == 'qwen3_5', model
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    assert shards and all((directory / p).is_file() and (directory / p).stat().st_size > 0 for p in shards)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'vocab.json', 'merges.txt'):
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected[f'models/Qwen3.5-4B/{name}'], (model, name)
    print(model, pins[model], 'index/shards present; frozen tokenizer hashes verified')
PY
```

以上检查刻意不比较 4B 与 9B 的模型 config 或来源回执哈希，它们应各自绑定正确模型。
两个固定版本的上述 tokenizer 文件在现有资产中相同；SFT 启动仍会渲染实际输入并核对
冻结 input_ids/labels，文件检查不能替代该校验。策略权重通过固定 HF revision 获取，
上述轻量检查只确认分片存在，不是对每个大权重文件做 SHA-256 校验。

## 下载共享量化用户模拟器

```bash
python -m tau3_grpo.models.download_simulator \
  --source huggingface --output "$TAU3_MODEL_ROOT/Qwen3.8-27B-AWQ-INT4"
```

该工具逐个核对17个文件的大小与 SHA-256，全部通过后才生成来源回执。
已验证文件会跳过；`.part` 可断点续传；已有不匹配文件会报错并保留，不静默覆盖。
下载源可换成 `--source hf-mirror` 或 `--source modelscope`，仍须通过同一固定文件清单的哈希。
ModelScope 使用镜像 master 地址，但不接受与固定 HF revision 不同的文件内容。

模拟器运行在独立的 Transformers 5.8.0 环境；策略训练保持固定训练环境，
不要把两者依赖混装。安装与启动细节见[双 A800 安装](setup-a800.md)和[模拟器说明](qwen38_simulator.md)。
下载后使用：

```bash
export TAU3_USER_MODEL="$TAU3_MODEL_ROOT/Qwen3.8-27B-AWQ-INT4"
export TAU3_USER_SERVED_MODEL_NAME=Qwen/Qwen3.8-27B-AWQ-INT4
export TAU3_USER_BASE_URL=http://127.0.0.1:8100/v1
```

这是路径和服务身份配置，尚未启动服务。比较4B与9B时保持同一模拟器、任务和采样协议。

## SFT、RL 与评测的模型身份

当前[三阶段 SFT](sft.md)的已冻结 profile 默认是4B配方；本页额外提供9B的资产准备，
不把4B训练预算或双卡GRPO历史结果声明为9B验收。9B正式实验需单独明确 profile、
模型选择、有效batch、更新预算和GPU资源验证，避免直接把4B模板当成9B已验证配方。

每个尺寸独立执行 A→B→C；A从对应基座开始，B/C通过 `SFT_MODEL_NAME_OR_PATH`
指向同尺寸前阶段选定的 merged model，不能跨尺寸混用adapter或基座。
GRPO/ARPO使用同一个已选定SFT模型作为各自起点，不使用另一算法训练后的模型作共同起点。
合并LoRA时绑定精确训练基座；评测使用实际导出的 merged checkpoint。
这些实验生成的权重由执行者训练、导出并记录，本页不提供尚未训练的新课程checkpoint下载地址。
