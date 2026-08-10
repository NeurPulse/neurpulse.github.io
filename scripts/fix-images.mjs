// 修复 Typora 误改的图片路径——每次用 Typora 编辑后，提交前跑一次
// 用法：npm run fix-images
import { readdirSync, readFileSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const notesDir = join(root, 'src', 'content', 'notes');
const assetsDir = join(notesDir, 'assets');

let fixed = 0;

for (const name of readdirSync(notesDir)) {
  if (!name.endsWith('.md')) continue;
  const file = join(notesDir, name);
  let content = readFileSync(file, 'utf8');
  const original = content;

  // Typora 把相对路径改成了 assets/ （每种图片类型各自修正）
  content = content.replace(
    /\(assets\//g,
    '(../../../public/images/math-modeling/'
  );
  // Typora 把 deep-learning 相对路径改成了本地绝对路径
  content = content.replace(
    /E:\\selfdocument\\web\\public\\images[\\/]deep-learning[\\/]/g,
    '../../../public/images/deep-learning/'
  );
  // Typora 把 math-modeling 相对路径改成了本地绝对路径
  content = content.replace(
    /E:\\selfdocument\\web\\public\\images[\\/]math-modeling[\\/]/g,
    '../../../public/images/math-modeling/'
  );
  // 从 QQ/微信拖进的图片（本地绝对路径）→ math-modeling
  content = content.replace(
    /\(D:\/QQ[^)]*\/([a-f0-9]{32}\.(?:png|jpg))\)/g,
    '(../../../public/images/math-modeling/$1)'
  );

  if (content !== original) {
    writeFileSync(file, content, 'utf8');
    console.log(`  已修复: ${name}`);
    fixed++;
  }
}

// 删除 Typora 自动创建的 assets 文件夹（图片已在 public/images/ 里）
if (existsSync(assetsDir)) {
  rmSync(assetsDir, { recursive: true });
  console.log('  已删除: src/content/notes/assets/');
}

console.log(fixed > 0 ? `\n共修复 ${fixed} 个文件，可以提交了。` : '\n所有文件路径正确，无需修复。');
