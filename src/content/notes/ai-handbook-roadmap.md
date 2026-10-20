---
title: 'AI 原理学习笔记（一）：零基础学习地图与使用方法'
description: '把机器学习、大模型、RAG、微调、Agent 与工程交付连成一条可循序学习的路线。'
pubDate: 2026-10-20
category: 'AI 技术与原理'
tags: ['学习路线', '人工智能', '零基础']
order: 1
featured: true
---

## 这套笔记怎样帮助我学习

这套笔记依据《AI 技术与原理宝典》整理，面向第一次系统学习 AI 的读者。这里的“理解”分成三步：先能用自己的话解释，再能跟着例子走一遍，最后能判断什么时候应该用它、什么时候应该换方法。

第一次读不要求会高数，也不要求有显卡。遇到公式，先看它在算什么，再看每个符号；遇到代码，先确认输入和输出，再追踪中间变量。涉及 PyTorch 的进一步实现会链接到已有专题。

全书原资料分为 19 篇，笔记按学习依赖重新拆成 28 篇。机器学习基础、神经网络、大模型、应用和工程都在路线里；已有的深度学习文章作为进一步展开的材料，不必一次读完。

## 1. 先看一个贯穿全系列的例子

假设公司有很多规章制度，我们想做一个“公司知识库助手”：

> 员工问“我入职两年，能休几天年假？”，系统找到现行制度，给出回答和出处。

这个目标看似只有一句话，却能串起大部分 AI 知识：

| 遇到的问题 | 对应知识 | 我应该学会回答什么 |
|---|---|---|
| 电脑怎么理解文字 | Token、Embedding | 文字怎样变成数字，数字的形状是什么 |
| 模型怎么组织回答 | Transformer、自回归生成 | 为什么一次预测一个 Token |
| 不知道公司最新制度 | RAG | 怎样先查资料，再根据资料回答 |
| 输出总是不符合格式 | Prompt、结构化校验 | 怎样说明任务并检查结果 |
| 想改变稳定的语言风格 | 微调 | 什么情况下值得修改参数 |
| 还想自动提交休假申请 | 工具、Workflow、Agent | 谁决定步骤，谁真正执行操作 |
| 回答偶尔出错 | 评估与失败分析 | 是没找到资料，还是找到却没读对 |
| 上线后很慢、很贵 | 缓存、显存、成本 | 时间和钱具体花在哪一步 |
| 员工查到了别人的合同 | 权限与安全 | 身份和数据权限怎样贯穿检索与工具 |

记住“问题先出现，技术才有位置”。不要为了用某个新名词，倒过来寻找需求。

## 2. AI、机器学习与大模型的关系

**人工智能 AI** 是一个很宽的目标：让计算机完成通常需要智能的任务。人写规则的专家系统也属于 AI。

**机器学习 ML** 是实现 AI 的一类方法：让程序根据样本调整参数，而不是为每一种情况手写规则。

**深度学习 DL** 是机器学习的一部分，用多层神经网络学习表示。例如从图像像素逐层学习边缘、纹理和物体特征。

**大语言模型 LLM** 主要处理语言相关任务。现代常见 LLM 基于深度神经网络训练，但“大模型”这个范围还包括图像、音频和多模态模型。

因此，判断一个系统时，要分别问：任务是什么？模型是什么？外围系统做了什么？一个聊天界面并不能告诉我们它有没有知识库、工具和长期记忆。

## 3. 按顺序阅读的完整目录

### 第一阶段：把基础概念变成可操作的知识

1. **本文：学习地图与使用方法**。
2. [最小数学与术语：样本、向量、概率、导数](/notes/ai-handbook-math-and-vocabulary/)。
3. [机器学习任务与模型：从规则到树模型、聚类和降维](/notes/ai-handbook-ml-model-selection/)。
4. [数据划分、预处理与防止泄漏](/notes/ai-handbook-data-and-validation/)。
5. [评估指标与统计可信度](/notes/ai-handbook-metrics-and-statistics/)。
6. [神经网络怎样学习，为什么走向 Transformer](/notes/ai-handbook-neural-network-and-transformer/)。

完成标志：给你一张表，你能指出样本、特征和标签；能解释为什么训练集高分不代表上线有效。

### 第二阶段：理解大模型怎样运行

7. [Token、Embedding 与语义相似度](/notes/ai-handbook-token-and-embedding/)。
8. [预训练、指令训练与逐步生成](/notes/ai-handbook-llm-training-and-generation/)。
9. [上下文窗口、Prefill、Decode 与 KV Cache](/notes/ai-handbook-context-and-kv-cache/)。
10. [长上下文、幻觉与多轮记忆](/notes/ai-handbook-long-context-and-hallucination/)。

完成标志：你能画出“文字 → Token → 隐表示 → 概率 → 新 Token”的流程，并区分参数里的知识与请求里提供的资料。

### 第三阶段：把模型接成应用

11. [Prompt 与上下文工程](/notes/ai-handbook-prompt-and-context-engineering/)。
12. [RAG：从文档到带出处的回答](/notes/ai-handbook-rag-pipeline/)。
13. [RAG 优化、混合检索、重排与 Graph RAG](/notes/ai-handbook-rag-optimization/)。
14. [全量微调、LoRA 与 QLoRA](/notes/ai-handbook-finetuning-and-lora/)。
15. [SFT、RLHF、DPO 与训练数据](/notes/ai-handbook-alignment-and-training-data/)。
16. [Function Call、MCP 与 Skill](/notes/ai-handbook-tools-mcp-and-skills/)。
17. [Workflow、Agent 与多 Agent](/notes/ai-handbook-workflow-and-agent/)。
18. [Agent 的状态、记忆与成本](/notes/ai-handbook-agent-memory-and-cost/)。
19. [多模态：图像、文字、语音与视频](/notes/ai-handbook-multimodal/)。
20. [应用评估：让“感觉不错”变成可检查的结果](/notes/ai-handbook-application-evaluation/)。

完成标志：面对一个需求，能解释为什么选择 Prompt、RAG、微调或工具，并能设计一个检查效果的小实验。

### 第四阶段：理解上线后的真实约束

21. [量化、显存与推理性能](/notes/ai-handbook-quantization-memory-and-performance/)。
22. [模型选型、Token 账单与 ROI](/notes/ai-handbook-model-selection-and-cost/)。
23. [云计算与网络：一次请求经过哪里](/notes/ai-handbook-cloud-and-network/)。
24. [认证、权限、接口与故障处理](/notes/ai-handbook-system-integration/)。
25. [综合实践：设计一个公司知识库助手](/notes/ai-handbook-knowledge-assistant-case/)。
26. [行业场景、POC 与方案交付](/notes/ai-handbook-industry-poc-and-delivery/)。
27. [安全、合规与持续运营](/notes/ai-handbook-safety-compliance-and-operations/)。
28. [产品分层、Harness 与前沿阅读方法](/notes/ai-handbook-product-framework-and-frontier/)。

完成标志：能画出系统架构，列出数据流、权限、成本和验收方法；出现坏结果时，知道从哪一层检查。

## 4. 每天怎样读

一次只选一个主题，用下面四步学习，不必追求读完多少页：

1. **读前提问**：这一篇要解决什么问题？我现在能给出什么猜测？
2. **跟一个例子**：手算一次，或者复制代码运行，再改一个输入观察变化。
3. **合上笔记解释**：尽量不使用原文术语，用自己的话讲给一个没学过的人听。
4. **回答自测**：先写答案，再展开参考答案；记下卡住的具体概念。

“看懂答案”和“自己能答出来”是不同的学习状态。答不出来时，把问题缩小，例如先解释为什么缓存 K、V，而不是直接要求自己解释整个推理引擎。

## 5. 哪些内容需要背，哪些不用背

需要留下的是**关系与判断方法**：训练与推理的差别、信息从哪里进入系统、指标的分母、公式的单位、不同方案的适用条件。

模型排行榜、云资源单价、软件版本和框架名单会变化。笔记里的金额与模型规模如果是教学假设，会明确注明；不能直接拿教学例子当报价或硬件承诺。

已有专题可以在需要时深入：

- [深度学习：线性回归](/notes/deep-learning-linear-regression/)帮助理解训练。
- [PyTorch：张量](/notes/pytorch-tensor/)帮助把数组写成代码。
- [Transformer：整体架构](/notes/transformer-overview/)展开注意力与模块连接。
- [CANN 与昇腾 NPU](/notes/cann-architecture-and-ascend-npu/)展开硬件与算子。

## 6. 第一次怎样运行代码

本系列的 16 段完整 Python 示例已在 Python 3.11.7 中实际运行，使用标准库，不需要显卡、模型下载、付费 API 或额外安装第三方包。它们演示核心计算或流程，不假装已经接上真实模型。

如果是第一次运行：

1. 打开终端，输入 `python --version`，确认命令能运行；部分 Windows 环境也可使用 `py --version`。若尚无 Python，可按 [Python 官方 Windows 使用说明](https://docs.python.org/3/using/windows.html)安装。
2. 新建一个文件，例如 `demo.py`，复制某一篇完整的 Python 代码块，保存为 UTF-8。不要复制网页上的围栏标记，也不要把终端命令混进 Python 文件。
3. 在该文件所在目录打开终端，输入 `python -X utf8 demo.py`。若使用 `py`，则输入 `py -X utf8 demo.py`。
4. 将打印结果与笔记里的注释比较。以 `#` 开头的内容是说明，不会执行。不要先连续复制多个彼此独立的示例到同一文件，避免变量相互覆盖。
5. 只改一个输入再运行，例如学习率、向量、Token 数或价格，观察结果为什么变。

`print(...)` 是把结果显示出来；`import` 是导入工具；`def` 定义一个可重复使用的函数；缩进表示哪些代码属于同一个步骤。出现 Traceback 时，从最后几行找报错类型与行号，再回到对应位置，不必把全部报错当成自己完全看不懂的问题。

JSON 代码块表示数据结构，text 代码块表示流程或输入示例，不应直接当作 Python 程序运行。

## 7. 复习卡片与自测

- 技术的起点：一个明确问题。
- 模型的作用：根据输入计算输出。
- 应用的作用：组织数据、模型、工具、权限和流程。
- 掌握的标准：能解释、能走例子、能判断边界。

**问题一：知识库问答为什么不一定要微调？**

<details><summary>参考答案</summary>

如果缺的是经常更新的公司资料，可以先检索，把资料放进当前请求。微调会修改参数，更新与验证的成本不同；是否需要它应由目标行为和评测决定。

</details>

**问题二：为什么看到一个聊天框，不能判断它是不是 Agent？**

<details><summary>参考答案</summary>

界面形式不能说明控制逻辑。关键要看系统是否根据目标和中间结果动态决定下一步，是否有工具、状态和停止条件。

</details>

## 原文定位与整理说明

主要依据：[《AI 技术与原理宝典》](https://essential1220.github.io/presales-tech-guide/)，“怎么读这本书”及第一至十九篇。使用的是 Zotero 于 2026-10-19 保存的快照。

本系列是按零基础学习顺序重新组织的解释与练习；代码中的小数据、金额和制度为教学构造，不是某家公司的真实信息。原文的“AI 产品成熟度六级”与“系统技术架构分层”会分别解释，避免把两种视角混成同一套标准。
