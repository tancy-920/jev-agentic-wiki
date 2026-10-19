# 归一化总览与RMSNorm

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

归一化调节特征尺度；统计轴、是否去均值及结构位置是三个独立选择。LayerNorm、RMSNorm和BatchNorm不能互换概念。

## 工作机制

LayerNorm对指定特征维计算当前样本均值与方差；RMSNorm使用均方根缩放，一般形式为x/√mean(x²+ε)*g，不进行均值中心化；BN按通道聚合批次及指定空间/序列位置。归一化位于子层前还是残差相加后，分别对应常见Pre-LN与Post-LN。

## 适用条件与易混淆点

RMSNorm省去中心化并不自动保证某硬件上更快。归一化参数量取决于是否有可学习bias和gain。mygpt中的峰值显存/时间属于具体实验，不能作为普遍收益。

## 关联知识

训练：优化器、学习率与数值精度、Decoder-only架构与mygpt实现、性能测量、正确性与适用条件、Pre-LN与Post-LN、BatchNorm：统计轴、训练与推理、LayerNorm：特征维归一化与BN/RMSNorm区别

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `pytorch-rmsnorm-2.11:9d22e35bf7b3:c0-1532` — [PyTorch 2.11 RMSNorm, captured public-source excerpt, captured excerpt L3134–L3160; snapshot L1–L27]
- `depth-1910.07467:14a6a097250f:c0-2733` — [Root Mean Square Layer Normalization, full page context, PDF file p. 1]
- `depth-2002.04745:8e1c3e3564f6:c14586-15771` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 4]
- `depth-2002.04745:8e1c3e3564f6:c4702-7341` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 2]
- `depth-2002.04745:8e1c3e3564f6:c7049-9310` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 2]
- `depth-1910.07467:14a6a097250f:c3581-5993` — [Root Mean Square Layer Normalization, full page context, PDF file p. 2]
- `depth-1910.07467:14a6a097250f:c7675-10103` — [Root Mean Square Layer Normalization, full page context, PDF file p. 3]
- `pytorch-rmsnorm-2.11:9d22e35bf7b3:c1267-2350` — [PyTorch 2.11 RMSNorm, captured public-source excerpt, captured excerpt L3157–L3175; snapshot L24–L42]
- `mygpt-model.py:e05006957de4:c2583-4229` — [mygpt / model.py, mygpt-model.py.py, L74–L128]
- `mygpt-model.py:e05006957de4:c1194-2694` — [mygpt / model.py, mygpt-model.py.py, L37–L76]
- `mygpt-results-rmsnorm_metrics.json:25396ba25fac:c0-1356` — [mygpt / results/rmsnorm_metrics.json, mygpt-results-rmsnorm_metrics.json.json, L1–L85]
