# 残差连接：恒等路径、梯度与维度

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

残差连接把子层变换与输入相加，让网络学习相对恒等映射的修正，而非每层都重新构造完整表示。

## 工作机制

设目标映射H(x)，学习F(x)=H(x)−x，输出y=x+F(x)。维度一致的identity shortcut无需额外投影参数；维度变化需兼容投影。形式上Jacobian为I+J_F，存在直接梯度项，但实际梯度还受堆叠、归一化及损失影响。若F接近0，残差块易接近恒等映射，这与ResNet缓解优化退化的动机相连。

## 适用条件与易混淆点

不能声称有残差就永不梯度消失或爆炸。Post-LN为LN(x+F(x))，其梯度还经过LN；Pre-LN为x+F(LN(x))。ResNet的图像实验与Transformer训练配置须分开归因。

## 关联知识

Transformer前向流程：输入、层堆叠与输出概率、Pre-LN与Post-LN、LayerNorm：特征维归一化与BN/RMSNorm区别、逐位置FFN / MLP：投影、激活与宽度

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `foundation-1512.03385:dace4e4f4f1a:c4287-6786` — [Deep Residual Learning for Image Recognition, full page context, PDF file p. 2]
- `foundation-1512.03385:dace4e4f4f1a:c2137-4287` — [Deep Residual Learning for Image Recognition, full page context, PDF file p. 1]
- `paper-1706.03762:eee8b5a0bcc2:c7110-8934` — [Attention Is All You Need, full page context, PDF file p. 3]
- `depth-2002.04745:8e1c3e3564f6:c14586-15771` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 4]
- `depth-2002.04745:8e1c3e3564f6:c4702-7341` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 2]
- `depth-2002.04745:8e1c3e3564f6:c7049-9310` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 2]
