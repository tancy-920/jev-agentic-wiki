# RoPE旋转位置编码

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

RoPE通过对query和key的成对通道施加位置相关旋转，将相对位置关系引入注意力分数。它不等于在输入上相加一个位置向量。

## 工作机制

位置m使用块对角旋转R_m，得到q_m=R_m q、k_n=R_n k。内积为q^T R_m^T R_n k。在固定频率旋转构造下，位置相关矩阵满足R_m^T R_n=R_(n−m)。位置因子R_(n−m)本身只由相对位移及固定的旋转构造决定，不含内容向量q、k；这一因子的性质不要求先固定q、k。完整分数q^T R_(n−m) k则同时依赖内容和相对位移：当只讨论完整分数随位置的变化时，需固定未旋转内容向量q、k及旋转构造，才把其余变化归于相对位移。固定q、k的条件约束的是这种完整分数的比较，不是位置因子成立的必要条件。不同内容即使距离相同也可以得到不同分数，因此不能概括为“注意力分数只由相对距离决定”。这是公式的直接推论，而非新增性能结论。旋转每个二维通道对保持其范数；频率与通道索引相关。

## 适用条件与易混淆点

推理缓存必须保证位置索引与旋转约定一致。相对位移结构并不保证任意长上下文准确；上下文扩展、频率缩放或插值是额外策略，不能混作原始RoPE。

## 关联知识

位置表示总览：绝对PE、RoPE与偏置、自注意力与 Q/K/V、KV Cache 增量解码

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `depth-2104.09864:abd4e80aae4d:c10919-13279` — [RoFormer: Enhanced Transformer with Rotary Position Embedding, full page context, PDF file p. 4]
- `depth-2104.09864:abd4e80aae4d:c14290-15871` — [RoFormer: Enhanced Transformer with Rotary Position Embedding, full page context, PDF file p. 5]
- `depth-2104.09864:abd4e80aae4d:c3293-6099` — [RoFormer: Enhanced Transformer with Rotary Position Embedding, full page context, PDF file p. 2]
- `depth-2104.09864:abd4e80aae4d:c5673-7829` — [RoFormer: Enhanced Transformer with Rotary Position Embedding, full page context, PDF file p. 2]
- `mygpt-readme.md:2d20f1a5d663:c14355-16474` — [mygpt / README.md, mygpt-readme.md.md, L239–L267]
