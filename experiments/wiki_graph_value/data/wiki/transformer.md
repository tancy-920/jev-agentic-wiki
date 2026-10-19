# Transformer 架构

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

Transformer用注意力连接序列位置，用逐位置前馈网络变换特征。原始模型是编码器—解码器，而不是所有Transformer都具有两套堆叠。

## 工作机制

原论文编码器与解码器各N=6层，d_model=512。编码器每层先多头自注意力，再FFN，每个子层均为LayerNorm(x+Sublayer(x))。解码器在因果自注意力后加入读取编码器输出的交叉注意力，再经过FFN。输入嵌入与位置表示相加；解码输出通过词表投影与softmax形成分布。

编码器中所有子层与embedding输出维度均为d_model=512，以便残差相加；自注意力和逐位置FFN不改变序列位置数量。FFN逐位置独立应用，位置之间共享该层参数。完整顺序是自注意力→残差相加→LayerNorm→逐位置FFN→残差相加→LayerNorm。这里的原始Post-LN次序不等于后续所有变体的次序。

## 适用条件与易混淆点

上述层数、宽度和Post-LN位置属于原论文设置，不应套给BERT、GPT或mygpt。注意力、FFN、残差和归一化承担不同作用，不能以其中一个解释全部训练稳定性。

## 关联知识

自注意力与 Q/K/V、交叉注意力与架构分类、Decoder-only架构与mygpt实现、归一化总览与RMSNorm、残差连接：恒等路径、梯度与维度、LayerNorm：特征维归一化与BN/RMSNorm区别、BERT MLM：输入遮盖与注意力可见性

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1706.03762:eee8b5a0bcc2:c8934-11426` — [Attention Is All You Need, full page context, PDF file p. 4]
- `author-t2t-v1.2.0-transformer:3ceeacf80adf:c7610-9993` — [Tensor2Tensor v1.2.0 transformer.py, author-t2t-v1.2.0-transformer.py: top-level definition, L213–L265]
- `depth-orca-osdi22:b0b993441062:c12952-15716` — [Orca: A Distributed Serving System for Transformer-Based Generative Models, full page context, PDF file p. 4]
- `paper-1706.03762:eee8b5a0bcc2:c7110-8934` — [Attention Is All You Need, full page context, PDF file p. 3]
- `paper-1910.10683:7aa53b0ecc3b:c132875-133573` — [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, full page context, PDF file p. 41]
- `hf-transformers-cache-practices-4.49.0:ca2ed3e79cc8:c7024-9618` — [Hugging Face Transformers v4.49.0 / Best Practices for Generation with Cache, Generate with Cache, L98–L117]
- `paper-1810.04805:6bb076ca96de:c10285-12186` — [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, full page context, PDF file p. 3]
- `paper-1706.03762:eee8b5a0bcc2:c11426-13973` — [Attention Is All You Need, full page context, PDF file p. 5]
- `mygpt-model.py:e05006957de4:c12523-14451` — [mygpt / model.py, mygpt-model.py.py, L319–L369]
- `mygpt-model.py:e05006957de4:c5047-6681` — [mygpt / model.py, mygpt-model.py.py, L151–L185]
- `mygpt-readme.md:2d20f1a5d663:c8442-10433` — [mygpt / README.md, mygpt-readme.md.md, L149–L175]
- `mygpt-readme.md:2d20f1a5d663:c12397-14611` — [mygpt / README.md, mygpt-readme.md.md, L193–L244]
- `foundation-pytorch-transformer-2.11:ebf4e5af7173:c47700-50426` — [PyTorch v2.11.0 transformer, foundation-pytorch-transformer-2.11.py, L1113–L1201]
