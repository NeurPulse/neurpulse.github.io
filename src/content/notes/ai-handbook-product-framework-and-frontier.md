---
title: 'AI 原理学习笔记（二十八）：产品分层、Harness 与前沿阅读方法'
description: '区分产品自主程度与系统技术分层，理解 Harness 的工程职责，并学会有证据地阅读不断变化的 AI 热点。'
pubDate: 2026-10-20
category: 'AI 技术与原理'
tags: ['AI 产品', 'Harness', '前沿', '知识框架']
order: 28
---

[返回系列导读](/notes/ai-handbook-roadmap/) · 先修：[Workflow 与 Agent](/notes/ai-handbook-workflow-and-agent/)、[系统集成](/notes/ai-handbook-system-integration/)

## 1. 为什么需要一个框架

听到“我们做了 AI 原生 Agent 平台”时，名字不能直接告诉我们系统有哪些能力。

框架的作用是帮助提问：运行时模型在哪里，能读什么、能做什么，谁决定下一步，谁验证结果，失败怎样处理。

它是一种分析工具，不是给产品贴先进或落后标签的标准。

## 2. 原资料中的六个产品层级

原资料第十八篇正文提出按能力与自主程度观察产品的六级框架。这里保留其思路并解释边界，**不把它当作行业统一标准**。

| 层级 | 主要能力 | 教学例子 |
|---|---|---|
| Prompt Wrapper | 包装输入与模型输出 | 标题改写工具 |
| Grounded / RAG | 根据外部证据回答 | 公司制度问答 |
| Tool-using AI | 调用外部能力 | 查询本人余额 |
| LLM Workflow | 在预设流程里组织模型与工具 | 审核后生成申请草稿 |
| Agentic Core | 根据目标与观察动态决定路径 | 为复杂目标补查信息、调整计划 |
| AI-native System | 将上下文、记忆、权限、运营与运行系统整合 | 持续服务真实业务的完整系统 |

一个产品可以同时有多种形态。更高自主程度不表示更适合所有任务；可控、简单的流程常能更好满足确定需求。

最后一级的“完整系统”也不是只有 Agent 才能具备：高质量 RAG 与 Workflow 同样需要权限、观测和运营。

## 3. 产品层级与技术架构分层不能混用

“算力—模型—数据—编排—应用—运营”是按系统职责拆模块；“Prompt Wrapper—RAG—工具—Workflow—Agent—完整系统”是观察产品能力与自主程度。

前者问“系统由什么组成”，后者问“运行时能承担什么”。一个第二级的 RAG 产品，也可能拥有完整的算力、数据、应用和运营模块。

原资料的导读与正文存在两种“六层”表述，学习时用这两个问题分开理解，就不会误以为“数据层就是 RAG 的同义词”。

## 4. 开发时用 AI 与运行时用 AI

用 AI 生成一个计算器网页，运行时仍可能只执行普通确定性代码。

网页上线后根据用户输入调用模型、检索资料并选择工具，则模型参与运行时。

这一区分用于说明架构与成本，不评价产品价值。一个完全确定性的计算器，可能正好比引入模型更适合精确计算。

## 5. Harness 是模型外围的工作环境

Harness 可以理解为让模型在任务中工作的运行框架与约束环境。它组织上下文、状态、工具、检查、预算与恢复。

例如编程任务里，它可能负责读取项目、保存进度、执行测试、管理工具权限、隔离代码运行、记录改动和验证结果。

它不是某一个统一协议，也不只是系统提示词。有关长任务状态与环境的工程案例可读 [Anthropic 的长任务 Harness 文章](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)；案例做法不等于所有任务的唯一设计。

## 6. 主要组件各自防什么

| 组件 | 解决的问题 |
|---|---|
| 任务与流程管理 | 目标丢失、步骤混乱 |
| 状态存储 | 中断后无法恢复、重复劳动 |
| 工具约束 | 名称、参数、范围与操作不受控 |
| 上下文管理 | 无关资料过多、关键条件被压掉 |
| 隔离与权限 | 错误动作影响不该接触的资源 |
| 自动检查 | 结果看似完成但实际不符合要求 |
| 观测与恢复 | 无法追踪、预算失控、故障后无法处理 |

外围约束可以减少后果、提高可检查性，但不提供“永远正确”保证。沙箱不能代替事实评估，测试也只能检查它覆盖的行为。

## 7. 先增强模型、上下文还是工具

如果正确资料没有进入系统，先改上下文与检索；如果需要精确计算或操作外部系统，先补工具；如果证据齐全、任务清楚却仍持续推理错误，再比较模型或训练。

这只是定位顺序的起点，还要通过实验验证。工具太多会增加选择负担，上下文太多会增加噪声，更强模型也可能更贵、更慢。

改善应对应具体失败，而不是“再加一个组件应该更强”。

## 8. 怎样阅读新产品与热点

原资料第十七篇包含具体 Agent 产品与工具的快照。学习时保留机制，使用前重新检查官方资料，避免把旧版本介绍当作当前事实。

每看一个新产品，都做一张“证据卡”：

1. 任务是什么，目标用户是谁？
2. 模型、工具、状态、权限与宿主分别在哪里？
3. 哪些能力已经可用，哪些只是演示或计划？
4. 测试使用什么数据、基线和样本量？
5. 失败、费用、延迟与安全边界是什么？
6. 资料发布于何时，功能与授权是否仍适用？

遇到“快十倍”，追问输入输出、硬件、并发、精度与测量口径。遇到“自主完成”，追问人工准备、提示、重试与验收条件。

## 9. FDE 与企业转型怎样理解

FDE，Forward Deployed Engineer，常被用来描述深入客户场景、把技术做成可运行方案的工程角色。具体公司职责不同，不是一套统一岗位标准。

其学习价值在于：懂模型还不够，还要读懂流程、数据、权限、验收和组织协作。

企业引入 AI 也不只是“买更强模型”。需要明确谁维护知识、谁判断错误、谁承担上线与恢复责任，以及收益如何测量。

## 10. 全系列术语回查

| 想复习的词 | 回看 |
|---|---|
| 参数、张量、概率、梯度、损失 | [最小数学](/notes/ai-handbook-math-and-vocabulary/) |
| 监督、聚类、树、SVM、PCA | [机器学习模型](/notes/ai-handbook-ml-model-selection/) |
| Precision、Recall、F1、MRR、NDCG | [指标与统计](/notes/ai-handbook-metrics-and-statistics/) |
| Token、BPE、Embedding | [文字与向量](/notes/ai-handbook-token-and-embedding/) |
| Prefill、Decode、KV、GQA、TTFT | [上下文与缓存](/notes/ai-handbook-context-and-kv-cache/) |
| RAG、召回、重排、Graph RAG | [RAG](/notes/ai-handbook-rag-pipeline/)与[优化](/notes/ai-handbook-rag-optimization/) |
| PEFT、LoRA、QLoRA、SFT、DPO | [微调](/notes/ai-handbook-finetuning-and-lora/)与[对齐](/notes/ai-handbook-alignment-and-training-data/) |
| MCP、Skill、Function Call、Agent | [工具](/notes/ai-handbook-tools-mcp-and-skills/)与[Agent](/notes/ai-handbook-workflow-and-agent/) |
| PTQ、QAT、显存、吞吐、TCO | [性能](/notes/ai-handbook-quantization-memory-and-performance/)与[成本](/notes/ai-handbook-model-selection-and-cost/) |
| SSO、鉴权、幂等、熔断、LLMOps | [集成](/notes/ai-handbook-system-integration/)与[运营](/notes/ai-handbook-safety-compliance-and-operations/) |

## 复习卡片与自测

- 框架帮助提问，不替代任务与证据。
- 产品自主程度和系统模块分层是两种视角。
- Harness 负责运行环境、状态、工具、检查与恢复。
- 前沿内容记录时间、条件、证据与限制。

**1. RAG 产品只有第二级，就不需要运营层吗？**

<details><summary>参考答案</summary>需要。第二级描述产品能力，不代表系统只有两种模块；文档更新、监控、权限与回滚仍然重要。</details>

**2. 什么情况下不该先换更强模型？**

<details><summary>参考答案</summary>例如问题来自缺资料、权限错误、错误工具或缓存过期。换模型不能直接修复这些环节。</details>

**3. 新产品演示很成功，我应该先问什么？**

<details><summary>参考答案</summary>问任务与测试条件、数据、基线、人工介入、失败样本、成本和版本，判断是否与你的场景可比。</details>

## 原文定位

依据：[原资料](https://essential1220.github.io/presales-tech-guide/) 第十七、十八篇、5.6。产品分级沿用原资料的学习视角，不作为行业统一标准或产品优劣排名。
