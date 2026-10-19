# 分页 KV Cache 与前缀共享

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

分页KV把逻辑连续序列映射到固定大小物理块，用块表管理KV，而不必为每条请求预留最大长度的连续区域。

## 工作机制

按需分配和非连续物理块可以消除外部碎片并缓解内部碎片；最后一个块仍可能未填满。共享前缀可引用相同物理块，写入时需正确引用计数与copy-on-write。分页影响缓存的分配、利用率和共享，不会单独减少每个独有token所需的K/V元素。

## 适用条件与易混淆点

不能说分页消除了全部内部碎片或减少了模型KV头数。分页与continuous batching是可配合但独立的内存管理/调度机制；本地原型不等同于完整vLLM系统。

## 关联知识

KV Cache 增量解码、Prefill 与 Decode、性能测量、正确性与适用条件、Continuous Batching与迭代级调度

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-2309.06180:7ba2ff703be1:c14989-17264` — [Efficient Memory Management for Large Language Model Serving with PagedAttention, captured public-source excerpt, captured excerpt L158–L166; snapshot L63–L71]
- `mygpt-paged_kv.py:4453b1b84e0f:c848-2277` — [mygpt / paged_kv.py, mygpt-paged_kv.py.py, L28–L72]
- `mygpt-paged_kv.py:4453b1b84e0f:c1976-3903` — [mygpt / paged_kv.py, mygpt-paged_kv.py.py, L65–L117]
- `mygpt-readme.md:2d20f1a5d663:c14355-16474` — [mygpt / README.md, mygpt-readme.md.md, L239–L267]
- `depth-orca-osdi22:b0b993441062:c22008-24173` — [Orca: A Distributed Serving System for Transformer-Based Generative Models, full page context, PDF file p. 6]
- `depth-orca-osdi22:b0b993441062:c23849-26414` — [Orca: A Distributed Serving System for Transformer-Based Generative Models, full page context, PDF file p. 6]
- `depth-2309.06180:4e592d02d9a3:c21000-23565` — [Efficient Memory Management for Large Language Model Serving with PagedAttention, full page context, PDF file p. 5]
- `depth-2309.06180:4e592d02d9a3:c23235-24761` — [Efficient Memory Management for Large Language Model Serving with PagedAttention, full page context, PDF file p. 5]
- `depth-2309.06180:4e592d02d9a3:c31560-34002` — [Efficient Memory Management for Large Language Model Serving with PagedAttention, full page context, PDF file p. 7]
- `depth-2309.06180:4e592d02d9a3:c33679-36093` — [Efficient Memory Management for Large Language Model Serving with PagedAttention, full page context, PDF file p. 7]
- `paper-2309.06180:7ba2ff703be1:c23494-24418` — [Efficient Memory Management for Large Language Model Serving with PagedAttention, captured public-source excerpt, captured excerpt L188–L190; snapshot L93–L95]
- `paper-2309.06180:7ba2ff703be1:c21043-23494` — [Efficient Memory Management for Large Language Model Serving with PagedAttention, captured public-source excerpt, captured excerpt L180–L187; snapshot L85–L92]
- `mygpt-readme.md:2d20f1a5d663:c8442-10433` — [mygpt / README.md, mygpt-readme.md.md, L149–L175]
- `mygpt-readme.md:2d20f1a5d663:c12397-14611` — [mygpt / README.md, mygpt-readme.md.md, L193–L244]
- `mygpt-results-bench_paged_kv.json:1ab4cce92ee9:c1337-2321` — [mygpt / results/bench_paged_kv.json, mygpt-results-bench_paged_kv.json.json, L80–L114]
- `mygpt-results-bench_paged_kv.json:1ab4cce92ee9:c0-1544` — [mygpt / results/bench_paged_kv.json, mygpt-results-bench_paged_kv.json.json, L1–L86]
