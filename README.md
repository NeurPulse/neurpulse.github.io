# 博客使用手册

我的个人学术博客：**https://neurpulse.github.io**

> 📖 写作时的注意事项（公式、图片、格式坑）看单独的 [写作指南.md](写作指南.md)

| 项目 | 信息 |
|---|---|
| 线上地址 | https://neurpulse.github.io |
| 代码仓库 | https://github.com/NeurPulse/neurpulse.github.io |
| 技术栈 | Astro 5（静态网站）+ KaTeX 数学公式 + Shiki 代码高亮 |
| 部署方式 | 推送代码到 GitHub 后自动构建发布（GitHub Actions） |
| 项目目录 | `E:\selfdocument\web` |

---

## 一、日常写作（最常用）

### 1. 写文章

**推荐方式——自动生成**（不用手敲 frontmatter）：

```bash
npm run new
```

按提示依次输入：类型、标题、文件名、分类、标签、是否精选。会自动生成带当天日期的完整文件并用 Typora 打开，直接写正文即可。

**手动方式**：

- **学习笔记** → 在 `src/content/notes/` 里新建 `.md` 文件
- **研究记录 / 论文笔记** → 在 `src/content/research/` 里新建 `.md` 文件

文件名决定网址，建议用英文小写加连字符，例如：
`src/content/notes/rough-volatility.md` → 网址 `/notes/rough-volatility/`

### 2. 文件开头必须有的格式（frontmatter）

```markdown
---
title: '文章标题'
description: '一句话描述（会显示在列表里）'
pubDate: 2026-08-07
category: '深度学习'
tags: ['LSTM', '时间序列']
---

正文从这里开始……
```

字段说明：

| 字段 | 必填 | 说明 |
|---|---|---|
| `title` | ✅ | 文章标题 |
| `description` | 建议 | 列表页显示的简介 |
| `pubDate` | ✅ | 格式 `YYYY-MM-DD`，列表按它倒序排列 |
| `category` | 建议 | 分类，笔记列表页会按它分组，不填默认"未分类" |
| `tags` | 可选 | 标签，可以有多个 |
| `draft` | 可选 | 写 `draft: true` 表示草稿，**不会发布到网站** |
| `featured` | 可选 | 写 `featured: true` 表示精选，**置顶显示在首页"精选文章"区** |

> 参考模板：`src/content/research/paper-reading-template.md`（论文精读模板，复制后改名即可用）

### 3. 发布到网站

在项目目录打开终端（Git Bash），执行三条命令：

```bash
git add .
git commit -m "添加文章：文章标题"
git push
```

等 **1~2 分钟**自动部署完成，刷新 https://neurpulse.github.io 就能看到。

部署进度可在 Actions 页面查看：
https://github.com/NeurPulse/neurpulse.github.io/actions
（绿色✓ = 成功；红色✗ = 失败，点进去能看到错误原因）

---

## 二、本地预览（可选，写长文时有用）

```bash
npm run dev
```

然后浏览器打开 http://localhost:4321 。**保存文件后页面自动刷新**，适合边写边看效果。按 `Ctrl + C` 停止。

其他命令：

| 命令 | 用途 |
|---|---|
| `npm run dev` | 写作预览（热更新） |
| `npm run build` | 检查网站能否正常构建（有错会提示） |
| `npm run preview` | 预览构建后的最终效果 |
| `npm install` | 换电脑/重装后，第一次必须先运行这个 |

---

## 三、Markdown 写作技巧

### 数学公式（KaTeX）

```markdown
行内公式：$dS_t = \mu S_t dt + \sigma S_t dW_t$

独立公式（单独占一行）：
$$
C = e^{-rT} \, \mathbb{E}\left[ \max(S_T - K, 0) \right]
$$
```

注意事项：
- 公式里的下划线 `_`、花括号 `{}` 直接写，不用转义
- **表格里如果用到 `|`（竖线），要写成 `\|`**，否则会被当成表格分隔符
- 公式太长时网页上会自动出现横向滚动条，不用担心

### 代码块

````markdown
```python
import numpy as np
```
````

开头 ``` 后面写语言名（python、cpp、matlab、bash 等）就有语法高亮。

### 插入图片

1. 把图片放进 `public/images/` 文件夹（可按文章建子文件夹，如 `public/images/deep-learning/`）
2. 文章里引用：

```markdown
![实验结果](/images/my-chart.png)
```

> 注意是 `/images/` 开头（正斜杠），不要写 `public/`。

**从 Typora 迁移文章时要特别注意图片**：

- ❌ `C:/Users/...` 这样的**本地绝对路径**——网站上不存在这些文件，必须先把图片复制到 `public/images/` 再改引用
- ❌ `http://` 开头的**外链图片**——网站是 https，浏览器会拦截 http 图片。解决办法同上：下载到 `public/images/` 本地化
- 图多的话直接找 Claude 帮忙批量处理，一句话的事

---

## 四、修改个人信息

| 想改什么 | 改哪个文件 |
|---|---|
| 网站名、作者名、首页研究方向 | `src/consts.ts` |
| 关于我页面（教育背景、邮箱等） | `src/pages/about/index.astro` |
| 项目展示列表 | `src/data/projects.ts` |

改完同样执行三条 git 命令发布。

---

## 五、内置功能说明

以下功能都是**全自动**的，写文章时无需任何额外操作：

- **文章目录**：根据正文里的 `##` 和 `###` 标题自动生成，显示在文章右侧（手机上看不到属正常，屏幕太窄时自动隐藏），滚动时会高亮当前小节
- **上一篇/下一篇**：文章底部自动按发布时间顺序生成
- **深色模式**：右上角太阳/月亮按钮切换，会记住你的选择；首次访问跟随系统设置
- **全文搜索**：导航栏"搜索"页，搜遍所有文章的标题和正文。注意：本地 `npm run dev` 时搜索不可用（索引是构建时生成的），用 `npm run build && npm run preview` 或线上网站测试
- **精选文章**：frontmatter 加 `featured: true` 即置顶首页，适合展示最有代表性的笔记

## 六、目录结构速览

```
web/
├── src/
│   ├── content/
│   │   ├── notes/         ← 学习笔记放这里（.md）
│   │   └── research/      ← 研究记录放这里（.md）
│   ├── pages/             ← 页面结构（一般不用动）
│   ├── components/        ← 网页组件（不用动）
│   ├── styles/global.css  ← 全站样式（改配色在这）
│   ├── consts.ts          ← 网站名、作者、研究方向
│   └── data/projects.ts   ← 项目列表
├── public/                ← 图片等静态文件
├── astro.config.mjs       ← 网站配置（一般不用动）
└── package.json
```

---

## 七、常见问题

**Q: 推送后网站没更新？**
1. 去 Actions 页面看是否部署成功（红叉就是失败了）
2. 确认文章的 `pubDate` 没写成未来日期太远
3. 浏览器强制刷新：`Ctrl + Shift + R`

**Q: 文章发布了但列表里看不到？**
检查 frontmatter 是否完整（尤其是 `pubDate`），`---` 必须顶格写；检查是不是写了 `draft: true`。

**Q: 命令行里 curl 自己网站打不开？**
正常现象。当前网络对 github.io 有限制，命令行不走代理所以打不开，**以浏览器为准**。

**Q: git push 报错 "Custom certificate bundle not found"？**
这台电脑修复过一次：证书在 `C:\Users\lichi\certs\ca-bundle.crt`。如果重装 Git 后复发，执行：
```bash
git config --global http.sslcainfo C:/Users/lichi/certs/ca-bundle.crt
```
（前提是那个证书文件还在；不在的话从 Git 安装目录 `mingw64/etc/ssl/certs/ca-bundle.crt` 复制一份过去）

**Q: 想绑定自己的域名（如 yourname.com）？**
买域名后在仓库 Settings → Pages → Custom domain 配置，再找 Claude 帮忙即可。

---

*本手册由 Claude 生成于 2026-08-07，网站有大的改动时记得更新。*
