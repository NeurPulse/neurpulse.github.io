interface ReadingStep {
  label: string;
  postId: string;
}

interface TopicGuide {
  id: string;
  aliases?: string[];
  summary: string;
  audience: string;
  prerequisites: string;
  outcome: string;
  steps: ReadingStep[];
}

// 固定专题锚点，新增或合并分类时不会改变已有分享链接。
export const TOPIC_GUIDES: Record<string, TopicGuide> = {
  'AI 技术与原理': {
    id: 'category-ai-handbook',
    summary: '面向零基础，从最小数学、机器学习与大模型机制，逐步学习 RAG、微调、Agent、评估与系统交付。',
    audience: '希望把 AI 术语、原理、例子与工程实践连起来的入门读者。',
    prerequisites: '可以从零开始；数学与术语会解释，最小代码主要使用 Python 标准库。',
    outcome: '理解模型怎样学习和生成，能区分不同应用方案，并检查数据、证据、权限、成本与验收。',
    steps: [
      { label: '学习地图与完整目录', postId: 'ai-handbook-roadmap' },
      { label: '最小数学与术语', postId: 'ai-handbook-math-and-vocabulary' },
      { label: '大模型训练与生成', postId: 'ai-handbook-llm-training-and-generation' },
      { label: 'RAG 与知识库', postId: 'ai-handbook-rag-pipeline' },
      { label: 'Agent 与流程', postId: 'ai-handbook-workflow-and-agent' },
      { label: '综合实践与交付', postId: 'ai-handbook-knowledge-assistant-case' },
    ],
  },
  '深度学习': {
    id: 'category-0',
    summary: '从回归与梯度下降出发，逐步学习卷积网络、自然语言处理和序列模型。',
    audience: '希望系统学习神经网络的入门读者。',
    prerequisites: 'Python 基础、矩阵运算与导数；代码部分会用到 PyTorch。',
    outcome: '理解模型训练的基本环节，并能沿着 CNN、RNN 与 LSTM 的脉络继续实践。',
    steps: [
      { label: '回归与训练基础', postId: 'deep-learning-linear-regression' },
      { label: '卷积网络', postId: 'deep-learning-complete-cnn' },
      { label: '序列模型', postId: 'deep-learning-rnn' },
    ],
  },
  PyTorch: {
    id: 'category-1',
    summary: '从张量和自动求导开始，整理数据读入、模型构建、初始化与损失函数。',
    audience: '准备把神经网络原理写成代码的读者。',
    prerequisites: 'Python 基础，了解数组与神经网络的基本概念。',
    outcome: '掌握数据、模型与损失函数的组织方式，进而了解自定义算子接入。',
    steps: [
      { label: '张量与求导', postId: 'pytorch-tensor' },
      { label: '模型构建', postId: 'pytorch-model-construction' },
      { label: '算子接入', postId: 'ascend-c-pytorch-kernel-integration' },
    ],
  },
  '数学建模': {
    id: 'category-2',
    summary: '从零基础数学与问题拆解开始，通过手算、可运行案例和练习答案，连接预测、优化、评价与实际决策。',
    audience: '希望理解每个知识点、知道何时使用，并逐步独立完成建模的入门读者。',
    prerequisites: '四则运算即可开始；矩阵、导数、概率和 Python 数组在基础篇补充。',
    outcome: '能定义变量、假设和约束，选择并验证模型，解释结果并完成从预测到备货的综合案例。',
    steps: [
      { label: '学习地图与完整目录', postId: 'mathematical-modeling-roadmap' },
      { label: '必备数学与 Python', postId: 'mathematical-modeling-foundations' },
      { label: '数据与统计基础', postId: 'mathematical-modeling-data-preprocessing' },
      { label: '规划与决策', postId: 'mathematical-modeling-optimization' },
      { label: '最优化绪论与课本习题详解', postId: 'mathematical-modeling-optimization-introduction' },
      { label: '时间序列与验证', postId: 'mathematical-modeling-time-series' },
      { label: '预测到备货综合案例', postId: 'mathematical-modeling-integrated-case' },
    ],
  },
  '分数布朗运动': {
    id: 'category-3',
    summary: '从协方差结构和 Hurst 指数理解分数布朗运动，认识两类模拟思路。',
    audience: '对随机过程与粗糙波动率建模感兴趣的读者。',
    prerequisites: '概率论、协方差与线性代数；模拟部分使用 NumPy。',
    outcome: '理解 Hurst 指数与增量相关性，并比较 Cholesky 分解和 FFT 模拟的思路。',
    steps: [{ label: '定义、性质与模拟', postId: 'fractional-brownian-motion' }],
  },
  '蒙特卡洛模拟': {
    id: 'category-4',
    summary: '以随机采样估计期望，并通过一个期权定价例子连接公式与 Python 实现。',
    audience: '想了解随机模拟如何解决数值问题的读者。',
    prerequisites: '概率与期望、正态分布，以及 Python / NumPy 基础。',
    outcome: '理解采样误差与置信区间，读懂欧式期权定价的最小模拟代码。',
    steps: [{ label: '随机采样到定价实现', postId: 'monte-carlo-basics' }],
  },
  'CANN / Ascend C': {
    id: 'category-6',
    aliases: ['category-5'],
    summary: '沿着硬件与算子基础、Kernel 实现、Tiling、融合、调试和性能分析逐步实践。',
    audience: '准备学习昇腾 NPU 自定义算子开发的读者。',
    prerequisites: 'C++ 与张量基础；运行代码需要对应的 CANN 开发环境。',
    outcome: '理解 Host / Device 分工、数据搬运与计算流水，并建立验证和调优的基本流程。',
    steps: [
      { label: '开发基础', postId: 'ascend-c-operator-development-basics' },
      { label: 'Add 算子实践', postId: 'ascend-c-add-operator-walkthrough' },
      { label: '调试与性能分析', postId: 'ascend-c-on-device-debugging' },
    ],
  },
  LLaMA: {
    id: 'category-7',
    summary: '从家族演进与整体架构切入，继续推导和实现 RMSNorm、Pre-Norm 与 RoPE。',
    audience: '了解 Transformer 后，希望继续学习 LLaMA 关键结构的读者。',
    prerequisites: '注意力机制、位置编码和 PyTorch 张量运算。',
    outcome: '把归一化、残差路径和旋转位置编码的原理与实现联系起来。',
    steps: [
      { label: '整体架构', postId: 'transformer-llama1-architecture' },
      { label: 'RMSNorm 实现', postId: 'transformer-rmsnorm-pytorch' },
      { label: 'RoPE 推导与实现', postId: 'llama-rope-relative-position' },
    ],
  },
  Transformer: {
    id: 'category-8',
    summary: '沿着注意力、位置编码、Encoder / Decoder 和训练推理，逐步组装 Transformer。',
    audience: '希望理解 Transformer 原理并追踪代码中张量形状的读者。',
    prerequisites: '矩阵乘法、Softmax、神经网络训练和 PyTorch 基础。',
    outcome: '理解 QKV、Mask 与模块组合，读懂训练和自回归推理的区别。',
    steps: [
      { label: '整体架构', postId: 'transformer-overview' },
      { label: '注意力机制', postId: 'transformer-self-attention' },
      { label: '代码组装', postId: 'transformer-pytorch-implementation' },
    ],
  },
};

export function getTopicId(category: string): string {
  return TOPIC_GUIDES[category]?.id
    ?? `topic-${Array.from(category, (character) => character.codePointAt(0)!.toString(16)).join('-')}`;
}

export function getTopicHref(category: string): string {
  return `/notes/#${getTopicId(category)}`;
}
