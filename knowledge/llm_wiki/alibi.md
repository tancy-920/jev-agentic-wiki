# ALiBi与相对位置偏置

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

ALiBi通过给注意力logit加与距离相关的线性偏置提供位置信号，与把位置向量加到输入不同。

## 工作机制

在因果位置j≤i上，可写bias(i,j)=−m_h*(i−j)，m_h是每头固定斜率，随后仍做softmax。T5相对偏置则按相对距离分桶，使用可学习标量加到logit，不是同一线性函数。T5原文设置32个桶，层间共享偏置参数而头间不同。

## 适用条件与易混淆点

ALiBi的训练长度外实验是指定模型、数据和长度下的观察，不保证无限外推。T5桶边界与最大距离依配置；不能拿某个测量的吞吐或困惑度当作所有相对位置方法的普遍结论。

## 关联知识

位置表示总览：绝对PE、RoPE与偏置、RoPE旋转位置编码、自注意力与 Q/K/V

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `foundation-2108.12409:cf6a3d07cc4d:c15367-17283` — [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, full page context, PDF file p. 5]
- `foundation-2108.12409:cf6a3d07cc4d:c16996-18966` — [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, full page context, PDF file p. 5]
- `foundation-2108.12409:cf6a3d07cc4d:c18966-21650` — [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, full page context, PDF file p. 6]
- `paper-1910.10683:7aa53b0ecc3b:c12742-15676` — [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, full page context, PDF file p. 5]
