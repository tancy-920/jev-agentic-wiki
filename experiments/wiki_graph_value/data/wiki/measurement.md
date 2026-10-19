# 性能测量、正确性与适用条件

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

性能结果必须与测量任务、硬件和配置绑定。正确性、延迟、吞吐、已分配显存和预留显存是不同指标。

## 工作机制

比较注意力实现时固定输入、权重、dtype、batch、头数与Tq/Tk，先核对输出容差，再做预热、GPU同步和重复计时，报告统计量及环境。推理需区分prefill/decode，服务需说明排队是否计入。torch.cuda的allocated与reserved含义不同，峰值统计应在明确边界重置。

## 适用条件与易混淆点

本库旧mygpt JSON不是当前硬件新测量。缺失硬件/重复次数/实际后端时应写未知，不能补造。缓存复用缩短实验时间也不能作为算法部署速度。

## 关联知识

KV Cache 增量解码、PyTorch SDPA 与后端选择、分页 KV Cache 与前缀共享、训练：优化器、学习率与数值精度

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `mygpt-bench.py:4ab37b20b5e7:c979-2847` — [mygpt / bench.py, mygpt-bench.py.py, L36–L89]
- `mygpt-bench.py:4ab37b20b5e7:c7703-9607` — [mygpt / bench.py, mygpt-bench.py.py, L243–L288]
- `mygpt-readme.md:2d20f1a5d663:c10199-12641` — [mygpt / README.md, mygpt-readme.md.md, L171–L195]
- `mygpt-readme.md:2d20f1a5d663:c6983-8692` — [mygpt / README.md, mygpt-readme.md.md, L128–L149]
- `mygpt-results-bench_speedup_vs_batch.json:0e40587d1e5c:c0-376` — [mygpt / results/bench_speedup_vs_batch.json, mygpt-results-bench_speedup_vs_batch.json.json, L1–L23]
- `mygpt-results-bench_forward_cost.json:f3529c0f7798:c0-284` — [mygpt / results/bench_forward_cost.json, mygpt-results-bench_forward_cost.json.json, L1–L22]
- `mygpt-requirements.txt:7d5067774196:c0-202` — [mygpt / requirements.txt, mygpt-requirements.txt.txt, L1–L7]
