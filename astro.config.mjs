// @ts-check
import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  // 部署后的真实地址
  site: 'https://neurpulse.github.io',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      // 双主题：配合深色模式，通过 CSS 变量切换
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
    },
  },
});
