// 原创教学流程图。运行 node scripts/generate-ai-handbook-diagrams.mjs 可重新生成。
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const destination = new URL('../public/images/ai-handbook/', import.meta.url);
await mkdir(fileURLToPath(destination), { recursive: true });
const escape = (s) => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const colors = { blue: ['#edf4ff', '#275a9b'], green: ['#eaf5ed', '#286947'], amber: ['#fff3df', '#8b5b15'] };

function render({ title, description, nodes, edges, labels = [], footer, height = 620 }) {
  const boxes = nodes.map(({ x, y, w = 230, h = 128, title: heading, lines, color = 'blue' }) => {
    const [fill, stroke] = colors[color];
    return `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${fill}" stroke="${stroke}" stroke-width="2"/>
    <text x="${x + 18}" y="${y + 35}" fill="${stroke}" font-size="23" font-weight="700">${escape(heading)}</text>
    ${lines.map((line, i) => `<text x="${x + 18}" y="${y + 69 + i * 27}" fill="#273244" font-size="19">${escape(line)}</text>`).join('')}</g>`;
  }).join('');
  const arrows = edges.map(({ d, dashed = false }) => `<path d="${d}" fill="none" stroke="#5c6c80" stroke-width="2.8" ${dashed ? 'stroke-dasharray="8 6"' : ''} marker-end="url(#arrow)"/>`).join('');
  const captions = labels.map(({ x, y, text }) => `<text x="${x}" y="${y}" fill="#526175" font-size="21">${escape(text)}</text>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 ${height}" role="img" aria-labelledby="diagram-title diagram-desc">
  <title id="diagram-title">${escape(title)}</title><desc id="diagram-desc">${escape(description)}</desc>
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#5c6c80"/></marker></defs>
  <rect width="1120" height="${height}" fill="#fff"/>
  <g font-family="system-ui, -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif">
  <text x="40" y="51" font-size="30" font-weight="700" fill="#172c46">${escape(title)}</text>
  <text x="40" y="85" font-size="19" fill="#526175">${escape(description)}</text>
  ${arrows}${boxes}${captions}
  <text x="40" y="${height - 27}" font-size="19" fill="#526175">${escape(footer)}</text>
  </g></svg>\n`;
}

const diagrams = {
  'training-loop.svg': {
    title: '一次训练：预测 → 比较 → 求梯度 → 更新',
    description: '输入与目标给出学习信号；参数在优化器这一步发生变化。',
    nodes: [
      { x: 40, y: 190, title: '1 前向预测', lines: ['输入经过各层', '得到预测值'] },
      { x: 310, y: 190, title: '2 计算损失', lines: ['预测与目标比较', '得到错误大小'] },
      { x: 580, y: 190, title: '3 反向传播', lines: ['沿计算图求梯度', '尚未更新参数'], color: 'amber' },
      { x: 850, y: 190, title: '4 优化器更新', lines: ['使用梯度与学习率', '改变可训练参数'], color: 'green' },
    ],
    edges: [{ d: 'M270 254H300' }, { d: 'M540 254H570' }, { d: 'M810 254H840' }, { d: 'M965 318V425H155V328' }],
    labels: [{ x: 40, y: 151, text: '数据、训练目标与当前参数' }, { x: 342, y: 461, text: '更新后的参数用于下一次前向计算' }],
    footer: '验证与测试使用独立数据检查效果，不因计算了一次损失就更新参数。', height: 550,
  },
  'llm-generation.svg': {
    title: '普通自回归推理：生成一个，再接着生成',
    description: '参数通常固定；每次新增的 Token 成为下一次预测的前缀。',
    nodes: [
      { x: 40, y: 185, w: 180, title: '输入文本', lines: ['规则、资料', '用户问题'] },
      { x: 255, y: 185, w: 180, title: '分词与向量', lines: ['Token ID', 'Embedding'] },
      { x: 470, y: 185, w: 180, title: '网络计算', lines: ['读取上下文', '产生隐表示'] },
      { x: 685, y: 185, w: 180, title: '候选分布', lines: ['logits', '采样或选最大'] },
      { x: 900, y: 185, w: 180, title: '新 Token', lines: ['追加到前缀', '检查停止条件'], color: 'green' },
    ],
    edges: [{ d: 'M220 249H245' }, { d: 'M435 249H460' }, { d: 'M650 249H675' }, { d: 'M865 249H890' }, { d: 'M990 313V425H345V323' }],
    labels: [{ x: 368, y: 463, text: '尚未结束：处理新增 Token，复用合适的 KV Cache' }],
    footer: '采样策略改变选择分布，不自动保证事实正确；停止后解码为可见文字。', height: 550,
  },
  'rag-pipeline.svg': {
    title: 'RAG：先建可检索资料，再按问题找证据',
    description: '上行是资料准备，下行是一次问答；来源、版本与权限必须贯穿流程。',
    nodes: [
      { x: 40, y: 155, title: '获取与解析', lines: ['文档、网页、扫描件', '检查文字与结构'] },
      { x: 310, y: 155, title: '分块与元数据', lines: ['保留条件与章节', '来源、时效、权限'] },
      { x: 580, y: 155, title: '计算表示', lines: ['文档块转向量', '可结合关键词索引'] },
      { x: 850, y: 155, title: '建立索引', lines: ['保存与查询入口', '支持更新与撤销'], color: 'green' },
      { x: 40, y: 425, title: '用户问题', lines: ['取得会话身份', '明确当前任务'] },
      { x: 310, y: 425, title: '检索与重排', lines: ['过滤可访问资料', '选取相关候选'] },
      { x: 580, y: 425, title: '组织证据', lines: ['问题＋资料＋约束', '保留引用标识'] },
      { x: 850, y: 425, title: '生成与检查', lines: ['检查答案与出处', '不足时追问或拒答'], color: 'green' },
    ],
    edges: [{ d: 'M270 219H300' }, { d: 'M540 219H570' }, { d: 'M810 219H840' }, { d: 'M965 283V355H425V415', dashed: true }, { d: 'M270 489H300' }, { d: 'M540 489H570' }, { d: 'M810 489H840' }],
    labels: [{ x: 40, y: 133, text: '建库' }, { x: 40, y: 403, text: '问答' }],
    footer: '检索没有找到正确证据时，先排查资料与召回；不要只要求生成器“更认真”。', height: 625,
  },
  'agent-loop.svg': {
    title: 'Agent：决策、行动、观察与停止',
    description: '动态选择下一步，但执行范围、预算和最终完成条件由系统共同约束。',
    nodes: [
      { x: 40, y: 165, title: '目标与状态', lines: ['已验证事实、约束', '待办与剩余预算'] },
      { x: 445, y: 165, title: '选择下一步', lines: ['回答、追问或工具', '依据当前观察'] },
      { x: 850, y: 165, title: '校验并执行', lines: ['权限、参数、白名单', '真实工具与回执'], color: 'amber' },
      { x: 445, y: 405, title: '观察并更新', lines: ['区分成功、失败', '核对实际产物'] },
      { x: 40, y: 405, title: '完成或停止', lines: ['目标完成或无进展', '预算、异常与人工'], color: 'green' },
    ],
    edges: [{ d: 'M270 229H435' }, { d: 'M675 229H840' }, { d: 'M965 293V469H685' }, { d: 'M560 405V303' }, { d: 'M445 469H280' }],
    labels: [{ x: 585, y: 363, text: '继续' }, { x: 312, y: 449, text: '退出' }],
    footer: '模型说“完成”只是待核实信号；有副作用的操作还需要真实授权与幂等控制。', height: 610,
  },
  'system-architecture.svg': {
    title: '知识库助手：模型之外还需要什么',
    description: '箭头是数据与调用关系；权限检查、日志与版本控制贯穿全部模块。',
    nodes: [
      { x: 40, y: 165, title: '用户与网关', lines: ['认证身份、限流', '接收问题与 trace ID'] },
      { x: 445, y: 165, title: '应用与编排', lines: ['选择检索和工具', '状态、模板、预算'] },
      { x: 850, y: 165, title: '模型服务', lines: ['读取证据与问题', '生成结构化结果'] },
      { x: 40, y: 405, title: '文档与索引', lines: ['原文、分块、向量', '版本、生效与权限'] },
      { x: 445, y: 405, title: '业务工具', lines: ['真实系统读写', '鉴权、幂等与回执'], color: 'amber' },
      { x: 850, y: 405, title: '验证与运营', lines: ['引用、效果、费用', '监控、反馈与回滚'], color: 'green' },
    ],
    edges: [{ d: 'M270 229H435' }, { d: 'M675 229H840' }, { d: 'M155 405V355H500V303' }, { d: 'M560 293V395' }, { d: 'M965 293V395' }, { d: 'M675 469H840', dashed: true }],
    footer: '资料更新、权限撤销与故障恢复需要覆盖索引、缓存、日志和工具产生的副本。', height: 610,
  },
};

for (const [name, specification] of Object.entries(diagrams)) {
  await writeFile(new URL(name, destination), render(specification), 'utf8');
}
console.log('Generated 5 original AI handbook diagrams.');
