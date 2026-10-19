# PyTorch SDPA 与后端选择

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

PyTorch scaled_dot_product_attention是注意力算子的接口，具体后端由设备、dtype、形状和支持条件等决定，不是FlashAttention的同义词。

## 工作机制

概念计算为softmax(QK^T*scale+bias)V，默认scale为1/√d_k。接口可选择math或可用融合后端；sdpa_kernel等控制可用实现。布尔attn_mask中True表示参与注意力，浮点mask作为logit偏置。dropout_p由调用者传入，评估时应显式传0。

## 适用条件与易混淆点

自动fallback不意味着必然发warning。文档所述无法运行融合实现的warning与用户禁用math、要求融合内核的情境有关。性能必须注明实际后端、硬件、dtype、batch、头数及Tq/Tk，不能把某次3.68倍当成接口保证。

## 关联知识

FlashAttention 与 IO 优化、分块注意力与在线 Softmax、性能测量、正确性与适用条件

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `pytorch-sdpa-2.11:42a07d742f7e:c3520-5776` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3203–L3222; snapshot L70–L89]
- `pytorch-sdpa-2.11:42a07d742f7e:c0-2094` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3134–L3170; snapshot L1–L37]
- `mygpt-model.py:e05006957de4:c1194-2694` — [mygpt / model.py, mygpt-model.py.py, L37–L76]
- `mygpt-model.py:e05006957de4:c8927-10267` — [mygpt / model.py, mygpt-model.py.py, L241–L267]
- `mygpt-bench.py:4ab37b20b5e7:c7703-9607` — [mygpt / bench.py, mygpt-bench.py.py, L243–L288]
- `mygpt-bench.py:4ab37b20b5e7:c9306-11043` — [mygpt / bench.py, mygpt-bench.py.py, L281–L329]
- `mygpt-results-bench_attention_impls.json:2a0918a1aa55:c4113-5673` — [mygpt / results/bench_attention_impls.json, mygpt-results-bench_attention_impls.json.json, L214–L286]
- `mygpt-results-bench_attention_impls.json:2a0918a1aa55:c2669-4331` — [mygpt / results/bench_attention_impls.json, mygpt-results-bench_attention_impls.json.json, L143–L224]
