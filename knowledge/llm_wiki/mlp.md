# 逐位置FFN / MLP：投影、激活与宽度

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

FFN又称Transformer中的MLP，对每个token独立做特征变换，位置之间共享同一组参数。它不负责像注意力那样混合不同位置。

## 工作机制

原始FFN(x)=max(0,xW1+b1)W2+b2，d_model→d_ff→d_model，论文基准宽度512→2048→512。第一层扩展特征，非线性后第二层投回残差所需维度。忽略bias时两矩阵参数约2*d_model*d_ff。门控FFN通常有两个输入投影和一个输出投影，参数比较必须配平中间宽度。

## 适用条件与易混淆点

不能把原始ReLU、mygpt的GELU和SwiGLU写成相同实现。忽略bias/乘加口径的估算应明确，参数相近不保证实测延迟相近。

## 关联知识

Decoder-only架构与mygpt实现、归一化总览与RMSNorm、FFN激活：ReLU、GELU与SwiGLU、残差连接：恒等路径、梯度与维度

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1706.03762:eee8b5a0bcc2:c11426-13973` — [Attention Is All You Need, full page context, PDF file p. 5]
- `mygpt-model.py:e05006957de4:c11035-12913` — [mygpt / model.py, mygpt-model.py.py, L288–L329]
- `depth-2002.05202:2f0b42e51459:c2216-3968` — [GLU Variants Improve Transformer, full page context, PDF file p. 2]
- `depth-2002.05202:2f0b42e51459:c3762-4507` — [GLU Variants Improve Transformer, full page context, PDF file p. 2]
- `paper-1706.03762:eee8b5a0bcc2:c7110-8934` — [Attention Is All You Need, full page context, PDF file p. 3]
- `paper-1706.03762:eee8b5a0bcc2:c8934-11426` — [Attention Is All You Need, full page context, PDF file p. 4]
- `mygpt-model.py:e05006957de4:c12523-14451` — [mygpt / model.py, mygpt-model.py.py, L319–L369]
- `mygpt-model.py:e05006957de4:c2583-4229` — [mygpt / model.py, mygpt-model.py.py, L74–L128]
- `mygpt-readme.md:2d20f1a5d663:c0-2086` — [mygpt / README.md, mygpt-readme.md.md, L1–L41]
- `mygpt-readme.md:2d20f1a5d663:c1849-3478` — [mygpt / README.md, mygpt-readme.md.md, L32–L66]
- `mygpt-readme.md:2d20f1a5d663:c14355-16474` — [mygpt / README.md, mygpt-readme.md.md, L239–L267]
