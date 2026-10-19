# 训练：优化器、学习率与数值精度

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

Transformer训练需同时考虑优化器、学习率变化、正则化和数值精度，某个稳定技巧的结论有模型及初始化范围。

## 工作机制

原论文使用Adam，β1=0.9、β2=0.98、ε=10^−9；学习率为d_model^−0.5*min(step^−0.5,step*warmup_steps^−1.5)，warmup_steps=4000，即先线性增长后按步数平方根倒数衰减。残差dropout与label smoothing等是不同正则手段。

## 适用条件与易混淆点

这些是原论文配置，不是所有Transformer的最优超参数。Pre-LN论文对初始化梯度及warmup的分析不保证任意大模型、优化器和学习率下可无条件移除warmup。混合精度和梯度缩放需结合具体实现核对。

## 关联知识

Decoder-only架构与mygpt实现、归一化总览与RMSNorm、性能测量、正确性与适用条件、Dropout：缩放、位置与推理配置、BatchNorm：统计轴、训练与推理

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1706.03762:eee8b5a0bcc2:c20388-21360` — [Attention Is All You Need, full page context, PDF file p. 7]
- `paper-1706.03762:eee8b5a0bcc2:c21360-23083` — [Attention Is All You Need, full page context, PDF file p. 8]
- `mygpt-train.py:52262ac90caa:c4587-6729` — [mygpt / train.py, mygpt-train.py.py, L126–L179]
- `mygpt-train.py:52262ac90caa:c6470-7904` — [mygpt / train.py, mygpt-train.py.py, L173–L210]
- `mygpt-readme.md:2d20f1a5d663:c1849-3478` — [mygpt / README.md, mygpt-readme.md.md, L32–L66]
- `mygpt-readme.md:2d20f1a5d663:c12397-14611` — [mygpt / README.md, mygpt-readme.md.md, L193–L244]
- `mygpt-readme.md:2d20f1a5d663:c14355-16474` — [mygpt / README.md, mygpt-readme.md.md, L239–L267]
- `mygpt-results-base_metrics.json:b0fb3bff29ed:c0-1363` — [mygpt / results/base_metrics.json, mygpt-results-base_metrics.json.json, L1–L85]
- `mygpt-results-gqa1_metrics.json:2a751af5abd5:c0-1360` — [mygpt / results/gqa1_metrics.json, mygpt-results-gqa1_metrics.json.json, L1–L85]
- `mygpt-results-gqa2_metrics.json:356f7ea90b77:c0-1359` — [mygpt / results/gqa2_metrics.json, mygpt-results-gqa2_metrics.json.json, L1–L85]
