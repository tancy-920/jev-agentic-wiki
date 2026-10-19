# FFN激活：ReLU、GELU与SwiGLU

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

FFN中的激活为线性投影引入非线性。ReLU、GELU与带门控SwiGLU属于不同计算结构。

## 工作机制

ReLU(x)=max(0,x)，GELU(x)=xΦ(x)。常见无bias SwiGLU(x)=[SiLU(xW)⊙(xV)]W2，SiLU(z)=z*sigmoid(z)。普通FFN有两个主要矩阵，门控版本有三个；为匹配参数/乘法预算，门控中间宽度常设为普通FFN的约2/3。

## 适用条件与易混淆点

2/3配平忽略bias、取整与实现开销，不保证相同实测延迟。GLU论文的质量结论限定其任务及训练预算；mygpt的GELU实现不能被描述为SwiGLU。

## 参数、算子和实测成本的比较

忽略bias，普通两矩阵FFN的主要参数是2*d_model*d_ff；三矩阵门控FFN是3*d_model*d_gated。配平得d_gated≈(2/3)*d_ff。若宽度保持不变，门控版本多一块矩阵；若按2/3缩窄，则不能据矩阵块数断言总参数、显存或耗时一定更大。逐元素门控、激活实现、kernel融合、形状和硬件都会影响实际成本，本文没有提供同硬件延迟或峰值显存测量，不能排名“ReLU总最省”或“SwiGLU普遍更快”。

论文例子在T5式编码器—解码器、d_model=768下，把普通d_ff=3072配平为门控2048。表1段填充任务524288步heldout对数困惑度为ReLU 1.677、GELU 1.679、SwiGLU 1.636；这是参数/计算匹配后的该任务观察，不是所有模型、数据或硬件上的普遍优劣。公式给出的ReLU/GELU数学性质与实测算子成本是不同层次，不能用“非光滑/平滑”直接推出性能排序。

## 关联知识

逐位置FFN / MLP：投影、激活与宽度、归一化总览与RMSNorm、Decoder-only架构与mygpt实现

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `depth-2002.05202:2f0b42e51459:c2216-3968` — [GLU Variants Improve Transformer, full page context, PDF file p. 2]
- `depth-2002.05202:2f0b42e51459:c3762-4507` — [GLU Variants Improve Transformer, full page context, PDF file p. 2]
- `paper-1706.03762:eee8b5a0bcc2:c11426-13973` — [Attention Is All You Need, full page context, PDF file p. 5]
- `mygpt-model.py:e05006957de4:c11035-12913` — [mygpt / model.py, mygpt-model.py.py, L288–L329]
- `mygpt-model.py:e05006957de4:c12523-14451` — [mygpt / model.py, mygpt-model.py.py, L319–L369]
- `foundation-pytorch-activation-2.11:8b8076c9383f:c1831-4154` — [PyTorch v2.11.0 activation, foundation-pytorch-activation-2.11.py, L87–L168]
- `foundation-pytorch-activation-2.11:8b8076c9383f:c19312-21442` — [PyTorch v2.11.0 activation, foundation-pytorch-activation-2.11.py, L702–L785]
- `foundation-1606.08415:54bf88ceea5f:c2784-4719` — [Gaussian Error Linear Units (GELUs), full page context, PDF file p. 2]
