---
title: 'AI 原理学习笔记（十六）：Function Call、MCP 与 Skill 分别是什么'
description: '追踪一次查年假工具调用，分清模型建议、宿主执行、连接协议、工具说明与真实权限。'
pubDate: 2026-10-20
category: 'AI 技术与原理'
tags: ['工具调用', 'MCP', 'Skill']
order: 16
---

[返回系列导读](/notes/ai-handbook-roadmap/) · 先修：[Prompt](/notes/ai-handbook-prompt-and-context-engineering/)

## 1. 会说“我查到了”不代表真的查过

语言模型可以生成“你的余额是 100 元”，但如果没有连接账户系统，它只是在输出文字。

工具把部分任务交给外部程序：查数据库、算数、读取文件、生成报表或提交业务操作。工具执行产生的结果，才是外部系统发生了什么的证据。

一次工具调用通常分成：提供工具说明 → 模型提出调用与参数 → 宿主校验权限并执行 → 返回结果 → 模型组织解释。

**模型输出的调用请求不等于操作已经完成。** 这一区别对发送、付款和写数据库尤其重要。

## 2. Function Call 是什么

可以先理解为：模型按照约定结构提出“调用哪个函数、传哪些参数”。

下面是教学用工具定义，不代表某家厂商 API 的原样格式：

~~~json
{
  "name": "get_leave_balance",
  "description": "查询当前已认证员工的可用年假，不修改数据",
  "input_schema": {
    "type": "object",
    "properties": {"year": {"type": "integer"}},
    "required": ["year"],
    "additionalProperties": false
  }
}
~~~

工具描述告诉模型何时使用，schema 约束参数形式。描述不清可能让模型选错；schema 合法也不表示请求有权限或年份合理。

## 3. 为什么身份不应该让模型随便填

如果工具允许传入任意员工 ID，却没有后端鉴权，用户可能让模型查询别人的余额。

更稳妥的方式是由登录会话确定身份，工具只接收必要业务参数。若确实需要代查，也要由服务器检查代理权限。

下面演示一个本地只读函数，固定会话用户，不连接真实人事系统：

~~~python
def get_leave_balance(arguments, session_user):
    if set(arguments) != {"year"}:
        raise ValueError("只接受 year")
    year = arguments["year"]
    if type(year) is not int or year != 2026:
        raise ValueError("教学数据只包含 2026 年")
    balances = {("employee-A", 2026): 3}
    key = (session_user, year)
    if key not in balances:
        return {"status": "not_found"}
    return {"status": "ok", "days": balances[key], "year": year}

result = get_leave_balance({"year": 2026}, session_user="employee-A")
print(result)  # {'status': 'ok', 'days': 3, 'year': 2026}
~~~

这段只演示服务器如何解释参数并绑定身份，没有实现登录、模型服务或实际 Function Call 协议。

## 4. 工具结果也需要结构化

不要让“查询失败”“没有该员工”和“余额为 0”都返回一个 0。它们对用户有完全不同的含义。

结果可包含状态、数据、证据 ID、时间和可重试信息。错误日志不应把密钥、内部堆栈或其他用户数据直接送给模型和用户。

写操作还需要幂等键和操作回执。收到超时不代表没执行，盲目重试可能重复扣款或重复发通知。

## 5. MCP 连接了哪些角色

MCP，Model Context Protocol，是 AI 应用与外部能力连接的一种标准协议。

可以先分清：

- **Host**：用户使用的 AI 应用，组织模型、连接与交互。
- **Client**：Host 中与一个 Server 通信的协议参与者。
- **Server**：向客户端提供工具、资源或相关能力的服务。

这不是“模型自己安装一个插件后就直接控制全世界”。宿主仍要管理连接、访问、执行与结果展示。角色说明可查 [MCP 架构文档](https://modelcontextprotocol.io/docs/learn/architecture)。

工具的发现与调用、输入输出格式由协议定义。具体版本的字段和传输方式会演进，学习笔记不应把某个旧版本请求样例当作永久格式。

## 6. Tools、Resources 与 Prompts

在 MCP 的相关概念中：

- Tools 是可调用的能力，例如查库存。
- Resources 提供上下文数据，例如文档内容。
- Prompts 提供可复用的提示组织方式。

它们是协议中的不同对象，不意味着所有 Server 都必须支持全部类型。实际能使用什么，要看服务能力与权限。

MCP 不代替模型服务协议，也不自动解决业务授权。它使接入方式更统一，但不保证每个工具都可靠、安全、无费用。

## 7. Skill 与 Workflow 又是什么

在本系列中，Skill 泛指可复用的任务知识包或操作说明，可能包含步骤、资源、脚本、约束和检查方法。不同产品对 Skill 的具体格式与加载机制可能不同。

例如“报销审核 Skill”可以说明怎样读发票、哪些字段要检查、何时转人工。它可能调用 MCP 工具，也可以只指导本地操作。

Workflow 是把多个步骤组织成流程。Agent 则可能根据中间结果动态选择下一步。

| 概念 | 核心问题 |
|---|---|
| Function Call | 模型怎样表达一次工具调用 |
| MCP | 应用怎样标准化连接外部能力 |
| Skill | 一类任务的知识和方法怎样复用 |
| Workflow | 已知步骤怎样编排 |
| Agent | 路径不完全预设时，下一步怎样决定 |

这些概念可以在同一系统里同时出现。

## 8. 工具安全与提示注入

网页、文件或工具结果可能包含恶意指令。模型能读到，不代表这些文字能改变系统权限。

后端应检查身份、参数、资源范围、操作类型与预算。高影响写操作可以根据业务设计设置明确确认或审批；只读查询与写入不要共享不必要的权限。

只在 Prompt 里写“不要越权”，不能代替服务器鉴权。工具的真实权限应当小于等于当前用户允许的业务范围。

## 复习卡片与自测

- 模型提出调用，宿主与服务执行操作。
- 参数合法、权限合法、执行成功是三件事。
- MCP 统一连接协议，不提供万能信任。
- Skill 存知识方法，Workflow 排步骤，Agent 决定动态路径。

**1. 工具调用 JSON 生成成功，能说申请已提交吗？**

<details><summary>参考答案</summary>不能。还需要真实执行和明确回执，必要时查询操作状态。</details>

**2. 为什么员工 ID 不应只听模型传入的值？**

<details><summary>参考答案</summary>参数可能被用户操纵或模型填错，后端必须根据登录身份和授权校验可访问对象。</details>

**3. MCP Server 返回的文档里写“忽略所有规则”，该怎么办？**

<details><summary>参考答案</summary>把它当不可信数据，不自动提升为指令；权限和执行边界仍由宿主与服务控制。</details>

## 原文定位

依据：[原资料](https://essential1220.github.io/presales-tech-guide/) 8.2、8.4、8.9、17.5、14.8。协议细节应使用 [MCP 官方规范](https://modelcontextprotocol.io/specification/)核对当前版本。
