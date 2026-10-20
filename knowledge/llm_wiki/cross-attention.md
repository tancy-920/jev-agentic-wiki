# 交叉注意力与架构分类

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

交叉注意力的query与key/value来自不同序列表示。原始Transformer用它把解码器状态连接到编码器输出。

## 工作机制

Q来自解码器上一子层，K、V来自编码器最终输出；分数矩阵形状为T_query×T_memory，两个长度可以不同。编码器自注意力、解码器因果自注意力和交叉注意力是不同子层，不能只看矩阵公式相同就混为一谈。

## 适用条件与易混淆点

Decoder-only模型不必包含独立交叉注意力。Q/K输入来源不同不等于可以随意共享投影参数；改变参数共享就是模型约束变化。

## 固定memory与K/V复用

encoder可以使用双向自注意力；完整编码后的encoder输出若保持固定，交叉注意力的K/V由这份固定memory经各自投影产生，decoder每一步更新Q时可以复用这些K/V。这是由交叉注意力的输入来源得到的机制推导，官方EncoderDecoderCache也分别管理self-attention与cross-attention缓存。它不表示对不断追加输入的双向encoder可以只追加各层K/V：如果memory本身发生变化，依赖它的缓存也需更新或重新计算。

## 关联知识

自注意力与 Q/K/V、Decoder-only架构与mygpt实现、Transformer 架构

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1706.03762:eee8b5a0bcc2:c11426-13973` — [Attention Is All You Need, full page context, PDF file p. 5]
- `author-t2t-v1.2.0-transformer:3ceeacf80adf:c7610-9993` — [Tensor2Tensor v1.2.0 transformer.py, author-t2t-v1.2.0-transformer.py: top-level definition, L213–L265]
- `foundation-pytorch-transformer-2.11:ebf4e5af7173:c47700-50426` — [PyTorch v2.11.0 transformer, foundation-pytorch-transformer-2.11.py, L1113–L1201]
- `paper-1910.10683:7aa53b0ecc3b:c52189-54691` — [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, full page context, PDF file p. 17]
- `paper-1706.03762:eee8b5a0bcc2:c7110-8934` — [Attention Is All You Need, full page context, PDF file p. 3]
- `paper-1910.10683:7aa53b0ecc3b:c33101-35632` — [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, full page context, PDF file p. 11]
- `paper-1810.04805:6bb076ca96de:c6145-8482` — [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, full page context, PDF file p. 2]
- `paper-1810.04805:6bb076ca96de:c14307-16636` — [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, full page context, PDF file p. 4]
- `hf-transformers-cache-strategies-4.57.0:d5dfb6b93c31:c13455-13911` — [Hugging Face Transformers v4.57.0 / KV cache strategies, Encoder-decoder cache, L198–L203]
- `hf-transformers-cache-strategies-4.57.0:d5dfb6b93c31:c0-752` — [Hugging Face Transformers v4.57.0 / KV cache strategies, Document preamble, L1–L16]
- `author-t2t-v1.2.0-common-attention:4d7013143123:c24182-26383` — [Tensor2Tensor v1.2.0 common_attention.py, author-t2t-v1.2.0-common-attention.py: top-level definition, L686–L749]

- `hf-transformers-cache-4.57.0:50131d48d593:c2433-4394` — [Hugging Face Transformers v4.57.0 / Caching，缓存不变性或固定memory机制核查入口]
