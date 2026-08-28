import { access, cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const sourceRoot = process.argv[2] ?? 'E:\\CANN';
const projectRoot = path.resolve(import.meta.dirname, '..');
const notesRoot = path.join(projectRoot, 'src', 'content', 'notes');
const imagesRoot = path.join(projectRoot, 'public', 'images', 'cann');

const notes = [
  ['1day', 'ascend-c-operator-development-basics.md'],
  ['1day', 'cann-architecture-and-ascend-npu.md'],
  ['1day', 'cann-operator-and-tensor-basics.md'],
  ['2day', 'ascend-c-kernel-and-spmd-basics.md'],
  ['2day', 'ascend-c-vector-programming-paradigm.md'],
  ['2day', 'ascend-c-add-operator-walkthrough.md'],
  ['2day', 'ascend-c-generalized-tiling.md'],
  ['2day', 'ascend-c-pytorch-kernel-integration.md'],
  ['2day', 'ascend-c-sigmoid-practice.md'],
  ['3day', 'ascend-c-matmul-fundamentals.md'],
  ['3day', 'ascend-c-matmul-high-level-api.md'],
  ['3day', 'ascend-c-fused-operator-concepts.md'],
  ['3day', 'ascend-c-cv-fused-operator-development.md'],
  ['4day', 'ascend-c-on-device-debugging.md'],
  ['4day', 'ascend-c-on-device-profiling.md'],
  ['4day', 'ascend-c-simulator-profiling.md'],
];

await mkdir(notesRoot, { recursive: true });
await mkdir(imagesRoot, { recursive: true });

for (const [index, [day, filename]] of notes.entries()) {
  const source = path.join(sourceRoot, day, 'notes', filename);
  const destination = path.join(notesRoot, filename);
  let markdown = await readFile(source, 'utf8');

  markdown = markdown
    .replace("category: '未分类'", "category: 'CANN / Ascend C'")
    .replace(/(tags:\s*\[[^\r\n]*\]\r?\n)/, `$1order: ${index + 1}\n`)
    .replaceAll('](./images/', '](/images/cann/');

  await writeFile(destination, markdown, 'utf8');
}

for (const day of ['1day', '2day', '3day', '4day']) {
  await cp(path.join(sourceRoot, day, 'notes', 'images'), imagesRoot, {
    recursive: true,
    force: true,
  });
}

let imageReferenceCount = 0;
for (const [, filename] of notes) {
  const markdown = await readFile(path.join(notesRoot, filename), 'utf8');
  if (markdown.includes('](./images/')) {
    throw new Error(`${filename} still contains a source-relative image path.`);
  }

  for (const match of markdown.matchAll(/!\[[^\]]*\]\((\/images\/cann\/[^)]+)\)/g)) {
    const publicPath = decodeURI(match[1]).replace(/^\//, '');
    await access(path.join(projectRoot, 'public', publicPath));
    imageReferenceCount += 1;
  }
}

console.log(
  `Imported ${notes.length} CANN notes and verified ${imageReferenceCount} image references from ${sourceRoot}.`,
);
