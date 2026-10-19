# KV Cache 增量解码

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

KV Cache在自回归解码中保留历史token各层的key和value，避免每一步重新计算整个前缀的K/V。它复用计算，但会占用随上下文增长的显存。

## 工作机制

prefill先处理提示序列并建立缓存。随后每一步只生成新token的Q/K/V，将新K/V追加，再用当前Q读取历史缓存。若每层相同，缓存张量字节约为2*L*B*T*H_kv*d_head*s，其中2代表K和V，s为每元素字节数。GQA/MQA减少H_kv；量化减少s；分页改善分配与共享，优化的是不同因素。

## 显存路线：内容数量、表示精度与存储利用率

- **内容数量**：MQA/GQA让多个query头共享KV头，减少实际保存的K/V元素数量H_kv，而不是简单丢弃已计算的缓存。其他条件相同，MQA把H个KV头减到1个，缓存大小相应变为原来的1/H。改变投影结构可能需要训练或uptraining，不能保证原checkpoint直接转换后质量不变。
- **表示精度**：量化降低每个元素的存储位数s，不等于减少元素数量。还需计入量化元数据、残留高精度缓存及转换开销；HF文档警告短上下文且显存充足时可能增加延迟。
- **存储利用率**：分页KV用块表把逻辑连续的缓存映射到非连续物理块，按需分配，减少为每请求预留最大长度带来的浪费；末块仍可能有未使用空间。共享前缀可复用物理块，降低重复存储。它不会单独改变模型KV头数或独有token对应的K/V元素数量。

这些路线可组合，但不能互相替代：由缓存容量公式和分页分配机制可推得，减少KV头并不自动消除预留浪费/碎片；分页并不自动缩小每个独有token的KV张量。区分逻辑张量容量、实际分配及共享后的物理占用，不能笼统说分页不减少显存或读取量。该比较是机制归纳，不是新速度实验。

## 适用条件与易混淆点

缓存不是消除所有注意力计算，每个新query仍读取历史K/V。上下文、位置偏移和mask必须一致；训练或非自回归全序列任务不应直接套用增量推理收益。batch很小时缓存管理开销可能抵消收益。

## 关联知识

Prefill 与 Decode、GQA 分组查询注意力、分页 KV Cache 与前缀共享、性能测量、正确性与适用条件

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `hf-transformers-cache-4.57.0:50131d48d593:c2433-4394` — [Hugging Face Transformers v4.57.0 / Caching, Official caching documentation, L40–L75]
- `hf-transformers-cache-practices-4.49.0:ca2ed3e79cc8:c1447-3980` — [Hugging Face Transformers v4.49.0 / Best Practices for Generation with Cache, What is Cache and why we should care?, L27–L43]
- `hf-transformers-cache-4.57.0:50131d48d593:c4136-6760` — [Hugging Face Transformers v4.57.0 / Caching, Official caching documentation, L68–L96]
- `hf-transformers-cache-practices-4.49.0:ca2ed3e79cc8:c7024-9618` — [Hugging Face Transformers v4.49.0 / Best Practices for Generation with Cache, Generate with Cache, L98–L117]
- `hf-transformers-cache-4.57.0:50131d48d593:c0-2563` — [Hugging Face Transformers v4.57.0 / Caching, Official caching documentation, L1–L43]
- `paper-1911.02150:292091205298:c0-1731` — [Fast Transformer Decoding: One Write-Head is All You Need, captured public-source excerpt, captured excerpt L38–L43; snapshot L1–L6]
- `paper-2305.13245:3ed3d65bc419:c4200-6482` — [GQA, KV头数量、容量及训练范围，L62–L73]
- `paper-2309.06180:7ba2ff703be1:c14989-17264` — [PagedAttention，非连续块、按需分配与末块空位，L158–L166]
- `hf-transformers-cache-strategies-4.57.0:d5dfb6b93c31:c10932-13356` — [HF Quantized cache，低精度、缓存开销与短上下文延迟警告]
- `mygpt-model.py:e05006957de4:c14266-16158` — [mygpt / model.py, mygpt-model.py.py, L366–L411]
- `mygpt-model.py:e05006957de4:c15869-17957` — [mygpt / model.py, mygpt-model.py.py, L403–L455]
- `mygpt-sample.py:a522fb11f829:c2447-3941` — [mygpt / sample.py, mygpt-sample.py.py, L72–L114]
- `mygpt-sample.py:a522fb11f829:c867-2738` — [mygpt / sample.py, mygpt-sample.py.py, L27–L79]
- `mygpt-results-bench_speedup_vs_batch.json:0e40587d1e5c:c0-376` — [mygpt / results/bench_speedup_vs_batch.json, mygpt-results-bench_speedup_vs_batch.json.json, L1–L23]
