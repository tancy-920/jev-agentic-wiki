# Padding Mask：形状、布尔语义与输出边界

> 状态：AI整理草稿，未获人工逐项确认。本文是完整知识词条，不是论文逐字原文。

## 定义与用途

Padding Mask通常用于屏蔽补齐序列中的无效key，使真实query不从padding位置读取信息。它与屏蔽padding query输出不同。

## 工作机制

MultiheadAttention的key_padding_mask通常为(B,S)，布尔True表示对应key被忽略；浮点值作为加性偏置。屏蔽key列后再softmax，才能将概率重新归一化。因果mask限制未来，padding mask限制无效key，两者可以共同约束可见性；合并后不一定仍可用纯causal优化提示，但因果约束没有消失。SDPA布尔attn_mask则True表示允许，语义相反。

## 适用条件与易混淆点

key被屏蔽不代表padding query行自动为0，损失和输出处理需单独控制。全屏蔽行的softmax数值行为依实现/后端而变，应实测和显式处理，不能统一保证NaN或统一保证0。

## 关联知识

因果掩码与位置、PyTorch SDPA 与后端选择、自注意力与 Q/K/V、BERT MLM：输入遮盖与注意力可见性

## 来源追溯

以下是整理依据与核查入口，不是逐句正确性认证。

- `author-pytorch-v2.11.0-functional:06f330c5ce12:c231374-232367` — [PyTorch v2.11.0 canonical mask and MHA implementation, _canonical_mask, L6192–L6218]
- `author-pytorch-v2.11.0-functional:06f330c5ce12:c248872-251239` — [PyTorch v2.11.0 canonical mask and MHA implementation, multi_head_attention_forward, L6607–L6668]
- `foundation-pytorch-activation-2.11:8b8076c9383f:c37704-40193` — [PyTorch v2.11.0 activation, foundation-pytorch-activation-2.11.py, L1273–L1297]
- `foundation-pytorch-transformer-2.11:ebf4e5af7173:c4493-7539` — [PyTorch v2.11.0 transformer, foundation-pytorch-transformer-2.11.py, L116–L207]
- `foundation-pytorch-transformer-2.11:ebf4e5af7173:c9582-11976` — [PyTorch v2.11.0 transformer, foundation-pytorch-transformer-2.11.py, L242–L284]
- `pytorch-sdpa-2.11:42a07d742f7e:c0-2094` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3134–L3170; snapshot L1–L37]
- `pytorch-sdpa-2.11:42a07d742f7e:c1869-3693` — [PyTorch 2.11 scaled_dot_product_attention, captured public-source excerpt, captured excerpt L3167–L3207; snapshot L34–L74]
