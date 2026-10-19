# GQA 分组查询注意力

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

分组查询注意力让多个query头共享一组key/value头，在MHA与MQA之间折中KV容量和模型质量。

## 工作机制

若query头数H_q、KV头数H_kv且H_q可被H_kv整除，每组H_q/H_kv个query头读取同一个KV头。缓存字节在其他条件相同时与H_kv成正比。H_kv=H_q退化为MHA，H_kv=1对应MQA。共享KV改变投影与缓存结构，而非简单删除已有缓存的一部分。

## 适用条件与易混淆点

重复展开KV张量可能削弱工程收益。缓存容量下降比例不是端到端提速比例；已有权重改成GQA还涉及训练或uptraining策略，不能保证直接转换后质量不变。

## 缓存内容数量与存储利用率的边界

减少KV头数降低实际存储的K/V元素数量，在其他维度与精度相同时减少缓存容量和加载数据量。GQA论文的MHA→MQA例子是H个KV头减到1个；GQA使用中间数量的KV头。这个结构变化本身不实现分页、按需分配或前缀共享，因此不能仅由头数减少推断预留浪费/碎片已消除。后者属于独立的缓存存储管理问题，可与GQA组合；参见分页KV词条。此边界是结构和分配机制的推论，不是论文给出的普遍提速保证。

## 关联知识

MHA 多头注意力、MQA 多查询注意力、KV Cache 增量解码

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-2305.13245:3ed3d65bc419:c4200-6482` — [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, captured public-source excerpt, captured excerpt L62–L73; snapshot L21–L32]
- `paper-2309.06180:7ba2ff703be1:c14989-17264` — [PagedAttention，分页按需分配与非连续物理块，L158–L166]
- `paper-2305.13245:3ed3d65bc419:c13133-15425` — [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, captured public-source excerpt, captured excerpt L117–L125; snapshot L76–L84]
- `mygpt-model.py:e05006957de4:c0-1426` — [mygpt / model.py, mygpt-model.py.py, L1–L42]
- `mygpt-model.py:e05006957de4:c3941-5197` — [mygpt / model.py, mygpt-model.py.py, L119–L153]
- `mygpt-results-bench_gqa_memory.json:12a3cf8406d0:c0-612` — [mygpt / results/bench_gqa_memory.json, mygpt-results-bench_gqa_memory.json.json, L1–L30]
- `mygpt-results-gqa1_metrics.json:2a751af5abd5:c0-1360` — [mygpt / results/gqa1_metrics.json, mygpt-results-gqa1_metrics.json.json, L1–L85]
- `mygpt-results-gqa2_metrics.json:356f7ea90b77:c0-1359` — [mygpt / results/gqa2_metrics.json, mygpt-results-gqa2_metrics.json.json, L1–L85]
