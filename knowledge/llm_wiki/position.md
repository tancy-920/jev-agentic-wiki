# 位置表示总览：绝对PE、RoPE与偏置

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

位置表示给注意力提供顺序信息。绝对向量相加、旋转Q/K及logit距离偏置是不同注入位置的方法。

## 工作机制

原始正弦编码为PE(p,2i)=sin(p/10000^(2i/d_model))，PE(p,2i+1)=cos(p/10000^(2i/d_model))，加到编码器和解码器输入嵌入。固定偏移可由每对正余弦的线性变换表达。可学习位置表通过训练获得位置向量；RoPE旋转Q/K；T5与ALiBi在注意力logit加入相对偏置。

## 适用条件与易混淆点

固定公式可计算训练长度外的位置，不等于模型在那些位置仍准确。可学习表在当前实现中受表长限制，扩展需说明策略。各方法的外推实验不能跨数据集、长度和模型直接比较。

## 关联知识

Decoder-only架构与mygpt实现、KV Cache 增量解码、RoPE旋转位置编码、Token Embedding、位置嵌入与输出权重共享、ALiBi与相对位置偏置

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1706.03762:eee8b5a0bcc2:c14608-17145` — [Attention Is All You Need, full page context, PDF file p. 6]
- `mygpt-model.py:e05006957de4:c14266-16158` — [mygpt / model.py, mygpt-model.py.py, L366–L411]
- `mygpt-model.py:e05006957de4:c12523-14451` — [mygpt / model.py, mygpt-model.py.py, L319–L369]
- `depth-2104.09864:abd4e80aae4d:c10919-13279` — [RoFormer: Enhanced Transformer with Rotary Position Embedding, full page context, PDF file p. 4]
- `depth-2104.09864:abd4e80aae4d:c14290-15871` — [RoFormer: Enhanced Transformer with Rotary Position Embedding, full page context, PDF file p. 5]
- `depth-2104.09864:abd4e80aae4d:c3293-6099` — [RoFormer: Enhanced Transformer with Rotary Position Embedding, full page context, PDF file p. 2]
- `depth-2104.09864:abd4e80aae4d:c5673-7829` — [RoFormer: Enhanced Transformer with Rotary Position Embedding, full page context, PDF file p. 2]
- `paper-1706.03762:eee8b5a0bcc2:c25848-27513` — [Attention Is All You Need, full page context, PDF file p. 9]
- `mygpt-readme.md:2d20f1a5d663:c0-2086` — [mygpt / README.md, mygpt-readme.md.md, L1–L41]
- `mygpt-readme.md:2d20f1a5d663:c1849-3478` — [mygpt / README.md, mygpt-readme.md.md, L32–L66]
- `mygpt-readme.md:2d20f1a5d663:c14355-16474` — [mygpt / README.md, mygpt-readme.md.md, L239–L267]
- `paper-1706.03762:eee8b5a0bcc2:c13665-14608` — [Attention Is All You Need, full page context, PDF file p. 5]
- `paper-1910.10683:7aa53b0ecc3b:c12742-15676` — [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, full page context, PDF file p. 5]
- `foundation-2108.12409:cf6a3d07cc4d:c15367-17283` — [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, full page context, PDF file p. 5]
- `foundation-2108.12409:cf6a3d07cc4d:c16996-18966` — [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, full page context, PDF file p. 5]
- `foundation-2108.12409:cf6a3d07cc4d:c18966-21650` — [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, full page context, PDF file p. 6]
