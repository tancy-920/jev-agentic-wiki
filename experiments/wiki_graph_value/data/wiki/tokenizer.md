# Tokenizer：BPE、WordPiece、Unigram与SentencePiece

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

Tokenizer把原始文本转为离散token ID；归一化、预分词、子词模型和后处理是不同阶段。SentencePiece是工具/处理框架，不是与BPE、Unigram同层次的唯一算法。

## 工作机制

BPE训练通常反复合并频繁相邻符号对。WordPiece常用按词表似然改善选择合并的解释，推理采用词表最长匹配；训练选择规则与推理切分规则不能混淆。Unigram从候选子词词表及概率模型出发进行裁剪/分词，子词正则化允许训练时采样切分。BERT使用WordPiece，Sennrich等工作将BPE用于NMT，SentencePiece支持BPE与Unigram等模型。

## 适用条件与易混淆点

不同实现的WordPiece训练细节不应冒充Google原始实现完全公开。mygpt是字符级实验，不因本库包含BPE知识就已使用子词词表。未知个人经验不能编造为事实。

## 关联知识

Token Embedding、位置嵌入与输出权重共享、Decoder-only架构与mygpt实现、训练：优化器、学习率与数值精度

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `depth-tokenizer-hf-4.57:8e310fb8d331:c752-1751` — [Hugging Face Transformers v4.57.0 / Summary of the tokenizers, Summary of the tokenizers, L17–L35]
- `depth-tokenizer-hf-4.57:8e310fb8d331:c8001-9978` — [Hugging Face Transformers v4.57.0 / Summary of the tokenizers, Byte-Pair Encoding (BPE), L140–L170]
- `depth-1808.06226:fa30b4a82307:c7462-9739` — [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing, full page context, PDF file p. 3]
- `depth-1808.06226:fa30b4a82307:c9555-11805` — [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing, full page context, PDF file p. 3]
- `depth-1508.07909:e8c1f2db5b4b:c10093-12489` — [Neural Machine Translation of Rare Words with Subword Units, full page context, PDF file p. 3]
- `depth-1508.07909:e8c1f2db5b4b:c12489-14823` — [Neural Machine Translation of Rare Words with Subword Units, full page context, PDF file p. 4]
- `depth-1804.10959:365099810758:c7131-9576` — [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates, full page context, PDF file p. 3]
- `depth-1804.10959:365099810758:c11015-13351` — [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates, full page context, PDF file p. 4]
- `mygpt-readme.md:2d20f1a5d663:c0-2086` — [mygpt / README.md, mygpt-readme.md.md, L1–L41]
- `depth-tokenizer-hf-4.57:8e310fb8d331:c12609-13933` — [Hugging Face Transformers v4.57.0 / Summary of the tokenizers, WordPiece, L213–L229]
- `foundation-bert-tokenizer-4.57:15f92ca3917a:c16977-19600` — [Transformers v4.57.0 BERT WordPiece tokenizer implementation, foundation-bert-tokenizer-4.57.py, L395–L472]
- `paper-1810.04805:6bb076ca96de:c12186-14638` — [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, full page context, PDF file p. 4]
- `paper-1706.03762:eee8b5a0bcc2:c18059-20718` — [Attention Is All You Need, full page context, PDF file p. 7]
