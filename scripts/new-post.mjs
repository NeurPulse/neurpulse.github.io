// 新建文章脚本：交互式生成带 frontmatter 的 Markdown 文件
// 用法：npm run new
import { createInterface } from 'node:readline/promises';
import { writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const rl = createInterface({ input: process.stdin, output: process.stdout });

const CATEGORIES = ['数学建模', '深度学习', 'PyTorch', 'Transformer', '论文精读', '未分类'];

// 本地时区的今天，格式 YYYY-MM-DD
function today() {
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${d.getFullYear()}-${m}-${day}`;
}

// YAML 单引号字符串转义
const yaml = (s) => `'${s.replace(/'/g, "''")}'`;

const typeAns = (await rl.question('类型  1=学习笔记  2=研究记录  [默认 1]: ')).trim();
const dir = typeAns === '2' ? 'research' : 'notes';

const title = (await rl.question('标题: ')).trim();
if (!title) {
  console.log('标题不能为空，已取消。');
  process.exit(1);
}

let slug = (await rl.question('文件名（英文小写+连字符，如 pytorch-autograd）: ')).trim();
if (!slug) {
  const d = new Date();
  slug = `post-${today()}-${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`;
  console.log(`未填写，自动生成文件名: ${slug}`);
}
if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(slug)) {
  console.log('文件名只能包含小写字母、数字和连字符，已取消。');
  process.exit(1);
}

console.log('分类:  ' + CATEGORIES.map((c, i) => `${i + 1}=${c}`).join('  '));
const catAns = (await rl.question('分类序号 [默认 未分类]: ')).trim();
const category = CATEGORIES[parseInt(catAns, 10) - 1] ?? '未分类';

const tagsAns = (await rl.question('标签（逗号分隔，可留空）: ')).trim();
const tags = tagsAns ? tagsAns.split(/[,，]/).map((t) => t.trim()).filter(Boolean) : [];

const featured = (await rl.question('设为精选文章？(y/N): ')).trim().toLowerCase() === 'y';

rl.close();

const content = `---
title: ${yaml(title)}
description: ''
pubDate: ${today()}
category: ${yaml(category)}
tags: [${tags.map(yaml).join(', ')}]
${featured ? 'featured: true\n' : ''}---

## 小节标题

正文从这里开始……
`;

const filePath = join(root, 'src', 'content', dir, `${slug}.md`);
if (existsSync(filePath)) {
  console.log(`文件已存在: ${filePath}，为避免覆盖已取消。`);
  process.exit(1);
}

mkdirSync(dirname(filePath), { recursive: true });
writeFileSync(filePath, content, 'utf8');
console.log(`\n已创建: src/content/${dir}/${slug}.md`);

// 用系统默认程序（Typora）打开新文件
spawn('cmd', ['/c', 'start', '', filePath], { stdio: 'ignore', detached: true }).unref();
