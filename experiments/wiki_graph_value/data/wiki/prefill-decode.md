# Prefill 与 Decode

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

自回归服务常分为prefill和decode：前者建立提示前缀状态，后者逐token延长输出。二者的并行度和瓶颈不同。

## 工作机制

prefill同时处理多个已知token并写入KV；decode通常每条请求每轮只有一个新query，需要读取已有K/V。提示长度、输出长度、batch和KV布局共同影响吞吐与单token延迟。TTFT侧重首token到达前的等待，后续token延迟与总吞吐反映不同体验。

## 适用条件与易混淆点

不能用prefill长序列SDPA测量直接解释batch1增量decode的收益。比较必须分开阶段，并计入队列、调度和缓存分配等系统成本；本文不提供通用加速倍数。

## 关联知识

KV Cache 增量解码、因果掩码与位置、分页 KV Cache 与前缀共享、Continuous Batching与迭代级调度

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `hf-transformers-cache-4.57.0:50131d48d593:c2433-4394` — [Hugging Face Transformers v4.57.0 / Caching, Official caching documentation, L40–L75]
- `hf-transformers-cache-4.57.0:50131d48d593:c4136-6760` — [Hugging Face Transformers v4.57.0 / Caching, Official caching documentation, L68–L96]
- `hf-transformers-cache-practices-4.49.0:ca2ed3e79cc8:c3979-6531` — [Hugging Face Transformers v4.49.0 / Best Practices for Generation with Cache, What is Cache and why we should care?, L43–L79]
- `hf-transformers-cache-practices-4.49.0:ca2ed3e79cc8:c7024-9618` — [Hugging Face Transformers v4.49.0 / Best Practices for Generation with Cache, Generate with Cache, L98–L117]
- `hf-transformers-cache-4.57.0:50131d48d593:c0-2563` — [Hugging Face Transformers v4.57.0 / Caching, Official caching documentation, L1–L43]
- `paper-1911.02150:292091205298:c0-1731` — [Fast Transformer Decoding: One Write-Head is All You Need, captured public-source excerpt, captured excerpt L38–L43; snapshot L1–L6]
- `mygpt-model.py:e05006957de4:c15869-17957` — [mygpt / model.py, mygpt-model.py.py, L403–L455]
- `mygpt-model.py:e05006957de4:c0-1426` — [mygpt / model.py, mygpt-model.py.py, L1–L42]
- `mygpt-paged_kv.py:4453b1b84e0f:c0-1099` — [mygpt / paged_kv.py, mygpt-paged_kv.py.py, L1–L38]
- `mygpt-paged_kv.py:4453b1b84e0f:c6418-8055` — [mygpt / paged_kv.py, mygpt-paged_kv.py.py, L180–L225]
