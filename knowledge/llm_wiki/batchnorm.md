# BatchNorm：统计轴、训练与推理

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

BatchNorm按特征/通道使用批次统计进行标准化，再用可学习γ和β缩放平移。它与对每个token特征维独立归一化的LN不同。

## 工作机制

y=γ*(x−μ)/√(σ²+ε)+β。PyTorch BatchNorm1d输入(N,C)或(N,C,L)，后者每通道统计跨N和L，而不是对每token的C维统计。默认训练使用当前批统计并更新running统计，eval使用running统计；track_running_stats=False时训练和eval都使用批统计。训练前向方差估计与running variance更新的估计口径还需区分。

## 适用条件与易混淆点

BN并非不能用于任何序列任务，但变长、padding和小批量会影响统计。原论文用internal covariate shift解释动机，不应把这当成唯一已证实的收益机制。

## 关联知识

LayerNorm：特征维归一化与BN/RMSNorm区别、归一化总览与RMSNorm、训练：优化器、学习率与数值精度

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `foundation-1502.03167:ff1b9f13b590:c12196-14541` — [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, full page context, PDF file p. 3]
- `foundation-pytorch-batchnorm-2.11:46b3c108a19d:c9509-12131` — [PyTorch v2.11.0 batchnorm, foundation-pytorch-batchnorm-2.11.py, L265–L316]
- `foundation-pytorch-batchnorm-2.11:46b3c108a19d:c11899-14252` — [PyTorch v2.11.0 batchnorm, foundation-pytorch-batchnorm-2.11.py, L314–L361]
- `foundation-1502.03167:ff1b9f13b590:c16449-18435` — [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, full page context, PDF file p. 4]
- `foundation-1607.06450:2e66a69eecc9:c5764-7454` — [Layer Normalization, full page context, PDF file p. 2]
