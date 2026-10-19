# 因果掩码与位置

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

因果掩码使自回归位置不能读取未来token。它约束注意力可见性，不提供位置向量本身。

## 工作机制

在方形训练序列中，位置i通常只允许j≤i。可以在未来位置logit加−∞后再softmax，使其权重为0。原始解码器还将目标输入右移，使预测目标不直接泄露。绝对位置、RoPE或相对偏置负责顺序信号，不能用因果mask完全代替。

## 适用条件与易混淆点

KV Cache中T_query与T_key可能不同，不能直接假定任意接口的非方形causal对齐方式。布尔True的含义依API而异：MultiheadAttention的布尔mask表示屏蔽，SDPA布尔attn_mask表示允许。

## 关联知识

自注意力与 Q/K/V、KV Cache 增量解码、位置表示总览：绝对PE、RoPE与偏置、Padding Mask：形状、布尔语义与输出边界、BERT MLM：输入遮盖与注意力可见性

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `author-pytorch-v2.11.0-functional:06f330c5ce12:c231374-232367` — [PyTorch v2.11.0 canonical mask and MHA implementation, _canonical_mask, L6192–L6218]
- `author-pytorch-v2.11.0-functional:06f330c5ce12:c248872-251239` — [PyTorch v2.11.0 canonical mask and MHA implementation, multi_head_attention_forward, L6607–L6668]
- `paper-1706.03762:eee8b5a0bcc2:c8934-11426` — [Attention Is All You Need, full page context, PDF file p. 4]
- `paper-1706.03762:eee8b5a0bcc2:c11426-13973` — [Attention Is All You Need, full page context, PDF file p. 5]
- `mygpt-model.py:e05006957de4:c6488-7889` — [mygpt / model.py, mygpt-model.py.py, L181–L216]
- `mygpt-model.py:e05006957de4:c7739-9025` — [mygpt / model.py, mygpt-model.py.py, L212–L242]
- `pytorch-sdpa-2.11:42a07d742f7e:c3520-5776` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3203–L3222; snapshot L70–L89]
- `pytorch-sdpa-2.11:42a07d742f7e:c0-2094` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3134–L3170; snapshot L1–L37]
- `hf-transformers-cache-4.57.0:50131d48d593:c0-2563` — [Hugging Face Transformers v4.57.0 / Caching, Official caching documentation, L1–L43]
- `hf-transformers-cache-4.57.0:50131d48d593:c2433-4394` — [Hugging Face Transformers v4.57.0 / Caching, Official caching documentation, L40–L75]
- `paper-1706.03762:eee8b5a0bcc2:c7110-8934` — [Attention Is All You Need, full page context, PDF file p. 3]
- `pytorch-sdpa-2.11:42a07d742f7e:c1869-3693` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3167–L3207; snapshot L34–L74]
