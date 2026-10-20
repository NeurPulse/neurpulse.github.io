---
title: 'AI 原理学习笔记（九）：上下文窗口、Prefill、Decode 与 KV Cache'
description: '追踪模型读取提示和生成新 Token 的两个阶段，解释 KV Cache 省了什么，以及怎样估算缓存显存。'
pubDate: 2026-10-20
category: 'AI 技术与原理'
tags: ['KV Cache', '大模型推理', '显存']
order: 9
---

[返回系列导读](/notes/ai-handbook-roadmap/) · 先修：[逐步生成](/notes/ai-handbook-llm-training-and-generation/)

## 1. 上下文窗口是一张有限的工作台

模型一次能处理的 Token 范围有限。放上工作台的内容可能包括系统说明、用户问题、对话历史、检索文档、工具结果，以及已经生成的回答。

具体服务可能分别限制输入、输出和总长度。不能只看一个宣传的窗口数字，就假设任意输入加任意输出都被允许。

窗口也不是长期硬盘。应用下一次若不把资料或状态送回来，模型未必还能使用它。

## 2. Prefill：先读完已知输入

假设提示已经有 1000 个 Token。这些 Token 都是已知的，网络可以在因果掩码下批量处理，建立各层表示与 K、V 缓存，并产生用于预测下一个 Token 的分数。

这个阶段叫 **Prefill**。长输入通常会增加首次响应前的工作量。

**TTFT，Time To First Token** 是从请求开始到第一个输出 Token 的时间。它还可能包含网络、排队、路由、检索和其他服务步骤，不能把它全归因于模型算矩阵。

## 3. Decode：每次接着生成

第一个输出 Token 被选定后，系统再把这个 Token 送进网络，计算下一个 Token。如此重复，直到停止。

这个阶段叫 **Decode**。常用 **TPOT，Time Per Output Token** 衡量后续平均每个 Token 的生成时间；有些测量会使用相邻输出的间隔，定义要写清。

提示很长、回答很短，与提示很短、回答很长，对两个阶段的负担不同。评测时应该给出输入和输出长度分布。

## 4. 为什么不必每次重算所有历史

在标准因果 Transformer 推理中，前面位置不能读取后面的新 Token。因此在参数固定、位置与上下文等条件一致时，历史位置已经算出的 K、V 可以复用。

设已经处理“我 喜欢”，准备生成下一个 Token。无需为了新的查询，再把“我”的 K、V 和“喜欢”的 K、V 全部从头计算。

每层大致做：

~~~text
已有缓存：[K1,K2,...,Kt] 与 [V1,V2,...,Vt]
新 Token：算出 q_new、k_new、v_new
把 k_new、v_new 追加到缓存
用 q_new 与全部有效 K 比较
按注意力权重汇总对应 V
~~~

**缓存避免重算过去的 K、V，但新查询通常仍要读取有效历史 K、V。** 所以历史越长，Decode 的注意力读取与缓存占用仍会增加。

为什么通常不缓存历史 Q 给下一步用？因为下一步要用的是新的查询，历史查询已经完成了当时的输出计算。

KV Cache 的用途与限制也可参照 [Transformers 缓存说明](https://huggingface.co/docs/transformers/cache_explanation)。

## 5. 算一次缓存大小

对常见保存标准 K、V 张量的结构，粗略缓存字节数为：

$$
M_{\mathrm{KV}}
\approx 2LBT H_{\mathrm{KV}}d_h s
$$

- 2：既存 K，也存 V。
- $L$：网络层数。
- $B$：同时驻留的序列数，此处假定等长。
- $T$：每条序列实际缓存的 Token 数。
- $H_{\mathrm{KV}}$：KV 头数。
- $d_h$：每个头的维度。
- $s$：每个元素所占字节数，如 FP16 为 2。

教学配置：32 层、1 条序列、2048 Token、8 个 KV 头、每头 128 维、FP16：

$$
2\times32\times1\times2048\times8\times128\times2
=268435456\ \text{bytes}=256\ \text{MiB}
$$

若其他条件不变，序列长度翻倍，缓存约翻倍；同时驻留 4 条同长度序列，约为 1 GiB。实际还受分块、碎片、量化元数据、共享前缀、并行布局和架构影响。

~~~python
layers, batch, tokens = 32, 1, 2048
kv_heads, head_dim, bytes_per_value = 8, 128, 2
memory = 2 * layers * batch * tokens * kv_heads * head_dim * bytes_per_value
print(memory)          # 268435456
print(memory / 2**20)  # 256.0 MiB
~~~

GB 使用 $10^9$ 字节，GiB 使用 $2^{30}$ 字节。估算时应注明，不要把二者混着报。

## 6. MHA、MQA 与 GQA 怎样影响缓存

MHA 常让每个查询头有相应 K、V 头。MQA 让多个查询头共享一组 K、V；GQA 介于两者之间，分组共享。

因此公式里要填 **KV 头数**，不是直接把查询头数代进去。如果从 32 个 KV 头改成 8 个，在其他假设不变时，缓存元素量变为四分之一。

这是架构与训练相关选择，不是对任何模型切换一个开关就能完全不影响效果。

## 7. PagedAttention、连续批处理与 FlashAttention

**PagedAttention**等缓存管理思路把 KV 分块管理，减少预留连续大块内存带来的浪费，并支持相应共享策略。它主要改善缓存管理，不是让注意力从此不读历史。

**连续批处理**在服务过程中动态加入新请求、移除已完成请求，避免整批都等最慢的请求结束。它要在利用率、吞吐、排队与单请求延迟之间取舍。

**FlashAttention**通过分块与融合减少中间结果在显存和片上存储之间的往返，不必显式存下完整注意力矩阵。它优化访存与实现方式，不代表稠密全注意力的所有计算都变成线性复杂度。

进一步阅读：[PagedAttention 论文](https://arxiv.org/abs/2309.06180)、[FlashAttention 论文](https://arxiv.org/abs/2205.14135)。

## 8. KV Cache、前缀缓存与答案缓存

| 缓存 | 保存什么 | 复用条件 |
|---|---|---|
| KV Cache | 当前序列各层 K、V | 推理上下文与实现允许复用 |
| 前缀缓存 | 可复用前缀的计算状态 | 常要求匹配的 Token 前缀及相容配置 |
| 答案缓存 | 已生成的最终结果 | 问题、资料版本、身份权限等满足业务条件 |

前缀缓存是否跨请求、是否打折计费，取决于具体服务。答案缓存尤其要包含权限和数据版本，避免把 A 用户有权看到的结果返回给 B。

## 复习卡片与自测

- Prefill 处理已知输入；Decode 逐步延续。
- KV 省重算，代价是驻留显存。
- KV 头数与序列长度是缓存估算的重要变量。
- 优化吞吐不一定同时降低每个请求的等待时间。

**1. 开 KV Cache 后，历史长度还影响推理吗？**

<details><summary>参考答案</summary>会。仍要保存并读取有效历史 K、V；这里只省掉对历史投影等的重复计算。</details>

**2. 8 个 KV 头的公式误填成 32 个，会怎样？**

<details><summary>参考答案</summary>在相同其他假设下，估算会放大四倍。</details>

**3. 用户感觉“等很久才开始，开始后很快”，先看哪个阶段？**

<details><summary>参考答案</summary>先拆 TTFT：网络、排队、检索、Prefill 等。不能只测后续 Token 的速度。</details>

## 原文定位

依据：[原资料](https://essential1220.github.io/presales-tech-guide/) 4.4、4.13、13.1—13.2。公式按标准 KV 张量结构教学估算，不覆盖所有特殊缓存架构。
