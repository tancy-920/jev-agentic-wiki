# 自注意力与 Q/K/V

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

自注意力让同一序列中的位置按内容相互读取信息。Q、K、V来自同一隐藏序列，但通常使用不同的可学习投影。

## 工作机制

对X∈R^(B×T×d)，单头Q=XW_Q、K=XW_K、V=XW_V。计算A=softmax(QK^T/√d_k+M)，再计算Y=AV。每个query对key位置归一化；value承载被加权的内容。注意力权重与最终输出不同。掩码M在softmax之前约束可见位置。

## 适用条件与易混淆点

自注意力不是固定平均，也不保证权重可直接解释为因果重要性。缩放的常见方差解释依赖坐标独立、零均值等假设，不能在任意相关分布下断言方差必为d_k。

## 关联知识

因果掩码与位置、交叉注意力与架构分类、MHA 多头注意力、Attention评分：Q/K、缩放与加性注意力、Padding Mask：形状、布尔语义与输出边界

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1706.03762:eee8b5a0bcc2:c8934-11426` — [Attention Is All You Need, full page context, PDF file p. 4]
- `paper-1706.03762:eee8b5a0bcc2:c11426-13973` — [Attention Is All You Need, full page context, PDF file p. 5]
- `paper-1706.03762:eee8b5a0bcc2:c7110-8934` — [Attention Is All You Need, full page context, PDF file p. 3]
- `mygpt-model.py:e05006957de4:c3941-5197` — [mygpt / model.py, mygpt-model.py.py, L119–L153]
- `mygpt-model.py:e05006957de4:c5047-6681` — [mygpt / model.py, mygpt-model.py.py, L151–L185]
