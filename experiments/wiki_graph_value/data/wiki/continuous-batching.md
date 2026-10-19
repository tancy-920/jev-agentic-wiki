# Continuous Batching与迭代级调度

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

连续批处理在生成迭代边界动态调整活动请求集合，让已结束请求退出、等待请求进入，区别于等待整批全部结束才换批。

## 工作机制

Orca的迭代级调度与selective batching针对生成任务不同算子的批处理特性进行组织。请求输出长度不同，固定整批会留下空闲位置；逐迭代调度可利用这些机会，但需要管理每请求的KV、位置与完成状态。

## 适用条件与易混淆点

动态调度不能单独降低单token固有计算量，收益受负载和延迟目标影响。它不等于分页KV，也不能根据本地有分页原型就声称已实现生产级连续批处理。

## 关联知识

分页 KV Cache 与前缀共享、Prefill 与 Decode、性能测量、正确性与适用条件

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `depth-orca-osdi22:b0b993441062:c22008-24173` — [Orca: A Distributed Serving System for Transformer-Based Generative Models, full page context, PDF file p. 6]
- `depth-orca-osdi22:b0b993441062:c23849-26414` — [Orca: A Distributed Serving System for Transformer-Based Generative Models, full page context, PDF file p. 6]
- `depth-orca-osdi22:b0b993441062:c643-3163` — [Orca: A Distributed Serving System for Transformer-Based Generative Models, full page context, PDF file p. 2]
- `depth-orca-osdi22:b0b993441062:c5343-8296` — [Orca: A Distributed Serving System for Transformer-Based Generative Models, full page context, PDF file p. 3]
- `mygpt-readme.md:2d20f1a5d663:c14355-16474` — [mygpt / README.md, mygpt-readme.md.md, L239–L267]
