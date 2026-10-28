# Transformer 架构分类与应用场景

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与分类

这里的 encoder-only、decoder-only 和 encoder-decoder 指 Transformer 的三类模型组织方式，不是同一 encoder-decoder 模型内两个子层的不同名字。选择架构时，先看任务需要整段输入的表示、从前缀继续生成，还是把一段输入转换成另一段输出；这些是典型用法，不是互斥的任务能力边界。

## Encoder-only：理解与判别

以 BERT 为代表，encoder-only 用双向自注意力编码完整输入，得到包含左右上下文的 token 表示，再接适合任务的输出头。BERT 原论文展示的下游用途包括句子/句对分类、自然语言推断、命名实体识别，以及预测答案起止位置的抽取式问答。它适合从已知输入提取表示、标签或答案片段的任务；原生双向编码器并不是 GPT 式逐 token 因果续写器。MLM 是预训练目标，不能把它直接当作这些下游任务本身。

## Decoder-only：语言建模与续写

GPT 式 decoder-only 以因果自注意力读取已有前缀，预测并逐步追加下一个 token，适用于自回归语言建模、文本续写和生成。给定提示后继续输出文本与这一机制相容；是否具备可靠的聊天、指令遵循或某项业务能力还依赖训练数据、训练目标及适配，架构名称本身不能保证。decoder-only 也可以经适配完成理解或分类任务，不应说只能生成或不能理解。这里的 decoder-only 不等同于原始 Transformer 中带交叉注意力、读取独立 encoder 输出的完整 decoder。

## Encoder-decoder：输入到输出的条件生成

encoder 先编码源输入，decoder 一边读取这份 encoder 表示，一边因果生成目标序列。原始 Transformer 在机器翻译中将源语言句子映射到目标语言句子；T5 的 encoder-decoder 将翻译、摘要、问答、分类等统一写成输入文本到目标文本。因而它适合有明确源输入和目标输出的条件序列生成，而不只是翻译；分类标签也能作为目标文本生成。输入和输出长度可以不同。

## 使用边界与选择依据

需要对整段输入输出标签/片段，可先考察 encoder-only；需要从前缀持续产生 token，可先考察 decoder-only；需要明确的输入到输出转换，可先考察 encoder-decoder。这是依据输入输出机制给出的任务匹配思路，不是实测性能排行榜，也不是某种架构总优于其他架构的结论。三类的可做任务有交集；具体效果、成本与延迟必须结合模型、数据、训练和部署条件评估。不能仅凭当下流行程度就判断某类已被淘汰。

## 关联知识

Transformer 架构、交叉注意力与架构分类、Decoder-only架构与mygpt实现、BERT MLM：输入遮盖与注意力可见性

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `paper-1810.04805:6bb076ca96de:c0-2284` — BERT，PDF p. 1：双向表示及句子/词元任务。
- `paper-1810.04805:6bb076ca96de:c1994-3955` — BERT，PDF p. 1：GPT 左到右架构与下游迁移。
- `paper-1810.04805:6bb076ca96de:c14307-16636` — BERT，PDF p. 4：左上下文 decoder 可用于文本生成。
- `paper-1706.03762:eee8b5a0bcc2:c0-2632` — Attention Is All You Need，PDF p. 1：encoder-decoder 与机器翻译。
- `paper-1706.03762:eee8b5a0bcc2:c5591-7110` — Attention Is All You Need，PDF p. 2：输入编码及自回归输出。
- `paper-1910.10683:7aa53b0ecc3b:c9253-11726` — T5，PDF p. 4：三类架构的语言建模、分类和片段预测用途。
- `paper-1910.10683:7aa53b0ecc3b:c6490-9198` — T5，PDF p. 3：翻译、摘要、分类和问答的 text-to-text 表达。
