# 开发与验证

只维护双A800训练方案；本地CPU CI用于快速验证。算法实现归tau3_grpo，脚本只作入口，硬件参数归configs。详见 [开发标准](architecture/development_standard.md)。

发布前检查：

```bash
python scripts/maintenance/check_lint.py
python -m tau3_grpo.integrations.vendor_inventory
python scripts/maintenance/check_cpu.py --suite core --report-dir results/maintenance/core --require-no-skips
```

core在干净声明依赖环境执行。benchmark在已有冻结数据和真实tokenizer的环境执行；verl在固定框架环境执行。使用同一report参数保存提交、源码哈希、依赖版本与JUnit。GPU条件跳过必须逐项报告，不能称全层通过。

GitHub自动执行core；benchmark和verl需要另附回执。不要删断言、替换真实tokenizer或扩大lint基线来制造通过。修改配置需核验完整include链、CLI和最终Hydra结果。
