# LayerNorm：特征维归一化与BN/RMSNorm区别

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

LayerNorm对每个样本在指定特征维上计算当前输入均值与方差，统计不依赖同batch其他样本。

## 工作机制

常见Transformer输入(B,T,D)，LN(D)对每个(B,T)位置的D个特征计算μ与σ²，再施加逐特征γ、β。训练与推理通常都用当前输入统计，不像默认BN读取running mean/variance。RMSNorm使用均方根缩放但不减均值；Pre-LN/Post-LN描述的是子层摆放。

## 适用条件与易混淆点

归一化轴由normalized_shape决定，不能断言任何LN总只沿最后一维。LN不保证所有训练问题自动解决；位置、epsilon、精度与初始化均可能影响行为。

## 关联知识

BatchNorm：统计轴、训练与推理、归一化总览与RMSNorm、Pre-LN与Post-LN、残差连接：恒等路径、梯度与维度

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `foundation-1607.06450:2e66a69eecc9:c5764-7454` — [Layer Normalization, full page context, PDF file p. 2]
- `foundation-pytorch-normalization-2.11:fe394c3aea14:c1630-3907` — [PyTorch v2.11.0 normalization, foundation-pytorch-normalization-2.11.py, L59–L126]
- `foundation-pytorch-normalization-2.11:fe394c3aea14:c3565-6055` — [PyTorch v2.11.0 normalization, foundation-pytorch-normalization-2.11.py, L120–L173]
- `paper-1706.03762:eee8b5a0bcc2:c7110-8934` — [Attention Is All You Need, full page context, PDF file p. 3]
- `depth-2002.04745:8e1c3e3564f6:c14586-15771` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 4]
- `depth-2002.04745:8e1c3e3564f6:c4702-7341` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 2]
