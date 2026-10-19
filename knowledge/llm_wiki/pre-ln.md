# Pre-LN与Post-LN

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

Pre-LN与Post-LN区别在子层归一化相对残差的位置，而不是是否存在残差。

## 工作机制

常见Post-LN为y=LN(x+F(x))，Pre-LN为y=x+F(LN(x))，堆叠末端还可能有final norm。Pre-LN的残差支路保留直接恒等项；Post-LN的归一化作用于相加结果。相关论文分析初始化时的梯度规模并在其设置中考察warmup需求。

## 适用条件与易混淆点

不能由这些结构式推出任何训练都不会梯度消失/爆炸，也不能把论文实验结论说成普遍无需warmup。归一化类型LN/RMSNorm与摆放Pre/Post是不同选择。

## 关联知识

归一化总览与RMSNorm、训练：优化器、学习率与数值精度、Transformer 架构

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `depth-2002.04745:8e1c3e3564f6:c14586-15771` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 4]
- `depth-2002.04745:8e1c3e3564f6:c4702-7341` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 2]
- `depth-2002.04745:8e1c3e3564f6:c7049-9310` — [On Layer Normalization in the Transformer Architecture, full page context, PDF file p. 2]
- `mygpt-model.py:e05006957de4:c11035-12913` — [mygpt / model.py, mygpt-model.py.py, L288–L329]
- `mygpt-model.py:e05006957de4:c12523-14451` — [mygpt / model.py, mygpt-model.py.py, L319–L369]
