# Decoder-only架构与mygpt实现

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

GPT式Decoder-only语言模型通过因果自注意力预测下一个token。mygpt是本地字符级受控实验实现，不代表完整生产推理系统。

## 工作机制

mygpt使用token与可学习位置嵌入相加，再经过重复的归一化、因果注意力、残差和MLP模块，最后投影到词表。代码可配置RMSNorm、KV Cache、KV头数和注意力实现等开关。训练可并行计算已知序列各位置，生成则逐步追加token。

## 适用条件与易混淆点

本地实现的层数、上下文和性能数字只适用于对应配置与原始JSON。不能把其他优化、推测解码或生产级连续批处理说成已经实现，也不能把小模型速度外推到大模型。

## 关联知识

位置表示总览：绝对PE、RoPE与偏置、逐位置FFN / MLP：投影、激活与宽度、归一化总览与RMSNorm、训练：优化器、学习率与数值精度、Tokenizer：BPE、WordPiece、Unigram与SentencePiece、Token Embedding、位置嵌入与输出权重共享

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1910.10683:7aa53b0ecc3b:c130789-133277` — [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, full page context, PDF file p. 41]
- `mygpt-readme.md:2d20f1a5d663:c0-2086` — [mygpt / README.md, mygpt-readme.md.md, L1–L41]
- `mygpt-model.py:e05006957de4:c12523-14451` — [mygpt / model.py, mygpt-model.py.py, L319–L369]
- `mygpt-model.py:e05006957de4:c14266-16158` — [mygpt / model.py, mygpt-model.py.py, L366–L411]
- `mygpt-readme.md:2d20f1a5d663:c12397-14611` — [mygpt / README.md, mygpt-readme.md.md, L193–L244]
- `mygpt-model.py:e05006957de4:c0-1426` — [mygpt / model.py, mygpt-model.py.py, L1–L42]
