# Dropout：缩放、位置与推理配置

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

Dropout训练时随机将部分激活置零，用于正则化；推理行为取决于具体接口及参数。

## 工作机制

常见inverted dropout中，保留概率为1−p，被保留激活乘1/(1−p)，因而在给定输入下期望保持。nn.Dropout的eval模式通常为恒等映射。原始Transformer在残差子层输出及嵌入与位置和等位置使用dropout；注意力概率dropout是另一个具体位置。

## 适用条件与易混淆点

不能把所有dropout接口都理解成由model.eval自动关闭。PyTorch SDPA总按传入的dropout_p应用dropout，调用者在评估时应显式传0；不同随机mask会影响正确性比较。

## 关联知识

训练：优化器、学习率与数值精度、PyTorch SDPA 与后端选择、Transformer前向流程：输入、层堆叠与输出概率

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `foundation-pytorch-dropout-2.11:f1abdbdda095:c0-2281` — [PyTorch v2.11.0 dropout, foundation-pytorch-dropout-2.11.py, L1–L79]
- `paper-1706.03762:eee8b5a0bcc2:c20388-21360` — [Attention Is All You Need, full page context, PDF file p. 7]
- `paper-1706.03762:eee8b5a0bcc2:c21360-23083` — [Attention Is All You Need, full page context, PDF file p. 8]
- `pytorch-sdpa-2.11:42a07d742f7e:c0-2094` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3134–L3170; snapshot L1–L37]
- `pytorch-sdpa-2.11:42a07d742f7e:c1869-3693` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3167–L3207; snapshot L34–L74]
