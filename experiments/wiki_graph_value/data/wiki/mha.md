# MHA 多头注意力

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

多头注意力在多个投影子空间中计算注意力，再拼接并投影回模型宽度。各头读取的位置权重可以不同。

## 工作机制

head_i=Attention(QW_Qi,KW_Ki,VW_Vi)，MultiHead=Concat(head_1,…,head_h)W_O。原论文采用h=8、d_k=d_v=64、d_model=512。固定总宽度时，拆分多个头不等于把每个头都设为完整模型宽度。标准MHA通常query与KV头数相同。

## 适用条件与易混淆点

更多头不必然更准确或更快。与GQA/MQA对比应固定模型宽度并交代query头、KV头、每头维度；不能仅用头数推断完整参数量。

## 关联知识

GQA 分组查询注意力、MQA 多查询注意力、自注意力与 Q/K/V

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1706.03762:eee8b5a0bcc2:c8934-11426` — [Attention Is All You Need, full page context, PDF file p. 4]
- `paper-1706.03762:eee8b5a0bcc2:c11426-13973` — [Attention Is All You Need, full page context, PDF file p. 5]
- `paper-2305.13245:3ed3d65bc419:c2100-4367` — [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, captured public-source excerpt, captured excerpt L50–L62; snapshot L9–L21]
- `paper-2305.13245:3ed3d65bc419:c0-2100` — [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, captured public-source excerpt, captured excerpt L42–L49; snapshot L1–L8]
- `mygpt-model.py:e05006957de4:c3941-5197` — [mygpt / model.py, mygpt-model.py.py, L119–L153]
- `mygpt-model.py:e05006957de4:c1194-2694` — [mygpt / model.py, mygpt-model.py.py, L37–L76]
