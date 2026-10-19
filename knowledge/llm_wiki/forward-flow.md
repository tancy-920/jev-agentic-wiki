# Transformer前向流程：输入、层堆叠与输出概率

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

前向流程把token ID变为上下文化表示，再变为输出分布。需先区分编码器—解码器与Decoder-only。

## 工作机制

原始编码器输入为缩放token嵌入加位置编码，经N层自注意力和FFN，每个子层残差后LN。解码器接收右移目标嵌入，依次经过因果自注意力、读取encoder输出的交叉注意力和FFN，再经线性词表投影及softmax。Decoder-only省去encoder与独立交叉注意力，以因果堆叠预测下一token。

## 适用条件与易混淆点

一次前向计算与生成循环不同：训练序列可并行计算多个位置，生成要逐步选择token、追加输入并维护缓存。原论文Post-LN与mygpt Pre-LN不能用同一个位置顺序描述。

## 关联知识

Transformer 架构、Tokenizer：BPE、WordPiece、Unigram与SentencePiece、Token Embedding、位置嵌入与输出权重共享、位置表示总览：绝对PE、RoPE与偏置、自注意力与 Q/K/V、逐位置FFN / MLP：投影、激活与宽度、Decoder-only架构与mygpt实现、Pre-LN与Post-LN

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1706.03762:eee8b5a0bcc2:c13665-14608` — [Attention Is All You Need, full page context, PDF file p. 5]
- `paper-1706.03762:eee8b5a0bcc2:c14608-17145` — [Attention Is All You Need, full page context, PDF file p. 6]
- `paper-1706.03762:eee8b5a0bcc2:c7110-8934` — [Attention Is All You Need, full page context, PDF file p. 3]
- `mygpt-readme.md:2d20f1a5d663:c0-2086` — [mygpt / README.md, mygpt-readme.md.md, L1–L41]
- `mygpt-model.py:e05006957de4:c12523-14451` — [mygpt / model.py, mygpt-model.py.py, L319–L369]
