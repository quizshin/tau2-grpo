# 配置入口

当前支持的双A800配置只从 `experiments/catalog.yaml` 进入。SFT/GRPO/ARPO/可选MT-GTPO共享组件，CLI要求显式profile。

目录中少量旧formal50、日期配方和A800硬件组件仍是奖励版本对照与回归测试的输入，不是新增支持环境；Python配置解析保留历史语义，不能把旧全参数多卡配方直接当成当前双卡入口。

5090/Paratera硬件入口已退出本分支，历史内容见archive/pre-a800-focus-20260929标签。

当前SFT仅列portable A109/B393/C500；旧v1课程启动依赖已清理，其训练配方保留为tests/fixtures中的回归数据，历史提交仍可查阅。
