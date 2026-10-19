# Token Embedding、位置嵌入与输出权重共享

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

Token Embedding将离散ID映射到可训练向量。查表输出、位置表示与输出词表投影有联系，但不是同一操作。

## 工作机制

ID形状(B,T)，词表矩阵(V,d_model)，查表得到(B,T,d_model)。原始Transformer将输入嵌入乘√d_model，再与同宽位置编码相加，并在其设置共享输入/输出相关嵌入权重。若共享词表矩阵，输出logit投影使用相应转置及具体实现的bias。

## 适用条件与易混淆点

√d_model缩放是原论文选择，方差解释需说明初始化与相关性假设，不能对任意embedding保证单位方差。权重共享须检查词表兼容和实际代码，不能从维度相同推断一定共享。

## 关联知识

Tokenizer：BPE、WordPiece、Unigram与SentencePiece、位置表示总览：绝对PE、RoPE与偏置、Decoder-only架构与mygpt实现、逐位置FFN / MLP：投影、激活与宽度

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1706.03762:eee8b5a0bcc2:c13665-14608` — [Attention Is All You Need, full page context, PDF file p. 5]
- `author-t2t-v1.2.0-modalities:b5f73ae48f67:c1130-3425` — [Tensor2Tensor v1.2.0 modalities.py, author-t2t-v1.2.0-modalities.py: top-level definition, L34–L107]
- `author-tensorflow-v1.3.0-init-ops:38d036411557:c8163-9257` — [TensorFlow v1.3.0 random initializer definitions, author-tensorflow-v1.3.0-init-ops.py: top-level definition, L247–L279]
- `mygpt-model.py:e05006957de4:c12523-14451` — [mygpt / model.py, mygpt-model.py.py, L319–L369]
- `mygpt-model.py:e05006957de4:c14266-16158` — [mygpt / model.py, mygpt-model.py.py, L366–L411]
- `paper-1706.03762:eee8b5a0bcc2:c14608-17145` — [Attention Is All You Need, full page context, PDF file p. 6]
