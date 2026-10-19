# FlashAttention 与 IO 优化

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

FlashAttention针对精确注意力的数据搬运开销进行优化，利用分块和片上存储减少HBM读写及中间矩阵物化。

## 工作机制

普通实现可能将T×T分数和概率写入显存。FlashAttention按Q/K/V块计算，结合在线softmax更新中间统计与输出，避免保存完整概率矩阵。原算法是精确注意力的IO优化，不是通过稀疏化或近似softmax省略任意关系。

## 适用条件与易混淆点

浮点累加顺序不同可产生容差内数值差异，精确算法不等于逐bit相同。二次注意力算术未因省去矩阵物化而自动变成线性；收益依序列长度、硬件与内核限制而变。

## 关联知识

PyTorch SDPA 与后端选择、分块注意力与在线 Softmax、性能测量、正确性与适用条件

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-2205.14135:b86de4ee5c52:c0-2471` — [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, full page context, PDF file p. 1]
- `paper-2205.14135:b86de4ee5c52:c3556-5723` — [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, full page context, PDF file p. 2]
- `pytorch-sdpa-2.11:42a07d742f7e:c3520-5776` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3203–L3222; snapshot L70–L89]
- `depth-2307.08691:f917fc544aae:c3449-6144` — [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, full page context, PDF file p. 2]
- `depth-2307.08691:f917fc544aae:c21985-24558` — [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, full page context, PDF file p. 9]
- `pytorch-sdpa-2.11:42a07d742f7e:c1869-3693` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3167–L3207; snapshot L34–L74]
- `mygpt-readme.md:2d20f1a5d663:c6983-8692` — [mygpt / README.md, mygpt-readme.md.md, L128–L149]
- `mygpt-readme.md:2d20f1a5d663:c8442-10433` — [mygpt / README.md, mygpt-readme.md.md, L149–L175]
