# 双 A800 环境契约

支持Linux/Python3.12、双A800、Qwen3.5-4B；固定依赖见qwen35-constraints.txt，安装入口setup_qwen35.sh。模拟器使用setup_qwen38_simulator.sh的隔离组件。CPU开发环境不等于GPU验收。

部署指南：[setup-a800](../docs/setup-a800.md)。vendor_patches.json及vendor_patch_notes.json记录固定上游与补丁；cpu_test_suites.json及lint_debt.json记录测试分层和历史静态债务。

旧5090、Torch2.8和多卡安装路线见archive/pre-a800-focus-20260929标签。历史测试仍依赖的离线工具已移至scripts/a800_research与scripts/engineering_checks，不是新增部署环境。
