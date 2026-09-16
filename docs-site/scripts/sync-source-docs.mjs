import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const docsRoot = path.resolve(__dirname, '..', 'docs');
const staticImgRoot = path.resolve(__dirname, '..', 'static', 'img', 'generated');
const repoRoot = path.resolve(__dirname, '..', '..');

const ensureDir = (dir) => mkdirSync(dir, { recursive: true });

const cleanDir = (dir) => {
  rmSync(dir, { recursive: true, force: true });
  ensureDir(dir);
};

const titleFromFile = (filename) =>
  filename
    .replace(/\.md$/i, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());

const normalizeMarkdown = (source, title) => {
  let content = readFileSync(source, 'utf8');
  content = content.replace(/\]\((reports|docs)\//g, '](/$1/');
  content = content.replace(/\]\(\/docs\/architecture\.md\)/g, '](/project-docs/architecture)');
  content = content.replace(/\]\(docs\/architecture\.md\)/g, '](/project-docs/architecture)');
  content = content.replace(/\]\(LICENSE\)/g, '](https://opensource.org/license/mit)');
  content = content.replace(/\]\((images|outputs)\//g, '](/img/generated/');
  content = content.replace(/<div align="center">/g, '');
  content = content.replace(/<\/div>/g, '');
  content = escapeMdxExpressions(content);

  if (!content.startsWith('---')) {
    content = `---\ntitle: ${title}\n---\n\n${content}`;
  }

  return content;
};

const escapeMdxExpressions = (markdown) => {
  const parts = markdown.split(/(```[\s\S]*?```)/g);
  return parts
    .map((part) => {
      if (part.startsWith('```')) return part;
      return part.replace(/\{/g, '&#123;').replace(/\}/g, '&#125;');
    })
    .join('');
};

const copyMarkdown = (source, targetDir, targetName = path.basename(source)) => {
  ensureDir(targetDir);
  const target = path.join(targetDir, targetName);
  writeFileSync(target, normalizeMarkdown(source, titleFromFile(targetName)));
};

const projectDocsDir = path.join(docsRoot, 'project-docs');
const reportsDir = path.join(docsRoot, 'reports');

cleanDir(projectDocsDir);
cleanDir(reportsDir);
cleanDir(staticImgRoot);

for (const file of ['README.md', 'README_COLAB.md', 'MASTER_EVALUATION_REPORT.md']) {
  const source = path.join(repoRoot, file);
  if (existsSync(source)) {
    copyMarkdown(source, projectDocsDir, file);
  }
}

for (const dirName of ['docs']) {
  const sourceDir = path.join(repoRoot, dirName);
  if (!existsSync(sourceDir)) continue;
  for (const file of readdirSync(sourceDir)) {
    if (file.endsWith('.md')) copyMarkdown(path.join(sourceDir, file), projectDocsDir, file);
  }
}

const sourceReports = path.join(repoRoot, 'reports');
if (existsSync(sourceReports)) {
  for (const file of readdirSync(sourceReports)) {
    if (file.endsWith('.md')) copyMarkdown(path.join(sourceReports, file), reportsDir, file);
  }
}

for (const imageDir of ['images', 'outputs']) {
  const sourceDir = path.join(repoRoot, imageDir);
  if (!existsSync(sourceDir)) continue;
  for (const file of readdirSync(sourceDir)) {
    if (/\.(png|jpg|jpeg|webp|svg)$/i.test(file)) {
      copyFileSync(path.join(sourceDir, file), path.join(staticImgRoot, file));
    }
  }
}

console.log('Synced README, docs, reports, and generated images into docs-site.');
