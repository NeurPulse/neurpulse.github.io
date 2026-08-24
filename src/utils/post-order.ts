type OrderedPost = {
  data: {
    order?: number;
    pubDate: Date;
    title: string;
  };
};

// 笔记总览中的专题顺序。没有列在这里的新分类会自动排在后面。
export const CATEGORY_ORDER = [
  '深度学习',
  'PyTorch',
  '数学建模',
  '分数布朗运动',
  '蒙特卡洛模拟',
];

export function compareCategories(a: string, b: string): number {
  const ai = CATEGORY_ORDER.indexOf(a);
  const bi = CATEGORY_ORDER.indexOf(b);
  if (ai === -1 && bi === -1) return a.localeCompare(b, 'zh-CN');
  if (ai === -1) return 1;
  if (bi === -1) return -1;
  return ai - bi;
}

// 优先使用 frontmatter 中明确设置的 order；旧文章没有 order 时仍能稳定排序。
export function compareReadingOrder<T extends OrderedPost>(a: T, b: T): number {
  const ao = a.data.order ?? Number.MAX_SAFE_INTEGER;
  const bo = b.data.order ?? Number.MAX_SAFE_INTEGER;
  if (ao !== bo) return ao - bo;

  const dateDiff = a.data.pubDate.valueOf() - b.data.pubDate.valueOf();
  if (dateDiff !== 0) return dateDiff;
  return a.data.title.localeCompare(b.data.title, 'zh-CN');
}
