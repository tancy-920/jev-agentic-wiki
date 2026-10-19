# 分块注意力与在线 Softmax

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

在线softmax用运行中的最大值和归一化量合并分块结果，使注意力不必保存完整分数矩阵。它不是逐块独立softmax后直接平均。

## 工作机制

已处理分块保留最大值m、指数和l、未归一化加权和o。新块最大值m_b、指数和l_b、加权和o_b到来时，m'=max(m,m_b)，l'=exp(m−m')l+exp(m_b−m')l_b，o'=exp(m−m')o+exp(m_b−m')o_b，最终输出o'/l'。旧块必须按新最大值重缩放。

## 适用条件与易混淆点

掩码和空块需专门处理，不能对全−∞行机械使用上述式子。Python教学分块代码不等同于融合GPU内核；减少峰值中间存储也不保证端到端更快。

## 关联知识

FlashAttention 与 IO 优化、PyTorch SDPA 与后端选择

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-2205.14135:b86de4ee5c52:c13633-15122` — [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, full page context, PDF file p. 4]
- `paper-2205.14135:b86de4ee5c52:c15122-16331` — [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, full page context, PDF file p. 5]
- `paper-2205.14135:b86de4ee5c52:c11869-13995` — [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, full page context, PDF file p. 4]
- `mygpt-model.py:e05006957de4:c10000-11173` — [mygpt / model.py, mygpt-model.py.py, L263–L289]
- `mygpt-model.py:e05006957de4:c3941-5197` — [mygpt / model.py, mygpt-model.py.py, L119–L153]
- `mygpt-results-bench_attention_impls.json:2a0918a1aa55:c1321-2882` — [mygpt / results/bench_attention_impls.json, mygpt-results-bench_attention_impls.json.json, L67–L154]
- `mygpt-results-bench_attention_impls.json:2a0918a1aa55:c0-1528` — [mygpt / results/bench_attention_impls.json, mygpt-results-bench_attention_impls.json.json, L1–L78]
