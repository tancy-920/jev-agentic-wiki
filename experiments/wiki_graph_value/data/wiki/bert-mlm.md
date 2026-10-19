# BERT MLM：输入遮盖与注意力可见性

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

BERT的Masked Language Modeling通过遮盖/替换部分输入token并预测原始token学习双向表示，不是给所有位置施加自回归三角mask。

## 工作机制

原论文随机选择15%的WordPiece位置作为预测目标；这些被选位置中80%替为[MASK]、10%替随机token、10%保持原token。损失只对被选预测位置计算。编码器仍可从左右上下文读取信息；输入替换与注意力连接掩码是两个不同层面的操作。

## 适用条件与易混淆点

15%选择率与80/10/10替换比例的分母不同，不能相加混用。MLM不能直接等同于GPT下一token目标；padding等有效性mask仍可存在，但不是为了实现MLM而对全部token做因果屏蔽。

## 关联知识

Padding Mask：形状、布尔语义与输出边界、因果掩码与位置、Tokenizer：BPE、WordPiece、Unigram与SentencePiece、Transformer 架构

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1810.04805:6bb076ca96de:c14307-16636` — [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, full page context, PDF file p. 4]
- `paper-1810.04805:6bb076ca96de:c0-2284` — [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, full page context, PDF file p. 1]
- `paper-1706.03762:eee8b5a0bcc2:c7110-8934` — [Attention Is All You Need, full page context, PDF file p. 3]
