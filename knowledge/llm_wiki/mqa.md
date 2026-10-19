# MQA 多查询注意力

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

多查询注意力保留多个query头，但所有query头共享单个K头和单个V头。它是KV头数为1的共享方案。

## 工作机制

在固定层数、batch、上下文、每头维度和dtype时，缓存由MHA的H_q个KV头降为1个，理论缓存容量比例为1/H_q。Q仍保留多个头，每个头的query可产生不同权重；共享KV不意味着只有一个注意力输出头。

## 适用条件与易混淆点

质量、内核支持与吞吐需实测。MQA不是分页或量化：它改变模型中的KV头数；分页改变内存组织，量化改变表示精度和每元素字节。

## 关联知识

MHA 多头注意力、GQA 分组查询注意力、KV Cache 增量解码

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1911.02150:292091205298:c0-1731` — [Fast Transformer Decoding: One Write-Head is All You Need, captured public-source excerpt, captured excerpt L38–L43; snapshot L1–L6]
- `paper-2305.13245:3ed3d65bc419:c4200-6482` — [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, captured public-source excerpt, captured excerpt L62–L73; snapshot L21–L32]
- `paper-2305.13245:3ed3d65bc419:c13133-15425` — [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, captured public-source excerpt, captured excerpt L117–L125; snapshot L76–L84]
- `mygpt-results-bench_gqa_memory.json:12a3cf8406d0:c0-612` — [mygpt / results/bench_gqa_memory.json, mygpt-results-bench_gqa_memory.json.json, L1–L30]
