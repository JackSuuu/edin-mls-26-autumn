import fs from 'node:fs/promises';
import path from 'node:path';
import { getDocument } from 'file:///C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdfjs-dist/legacy/build/pdf.mjs';

const slideRoot = 'D:/Desktop/uni_of_edinburgh/year4/machine_learning_systems/slides';
const outRoot = 'D:/Desktop/uni_of_edinburgh/year4/machine_learning_systems/cw/edin-mls-26-autumn/tmp/pdfs/course-slides-review';
await fs.mkdir(outRoot, { recursive: true });
for (const name of ['week7rag', 'week2-small_0', 'week8data-l3']) {
  const data = new Uint8Array(await fs.readFile(path.join(slideRoot, name + '.pdf')));
  const pdf = await getDocument({ data, useSystemFonts: true, disableFontFace: true }).promise;
  const pages = [];
  for (let index = 1; index <= pdf.numPages; index++) {
    const page = await pdf.getPage(index);
    const content = await page.getTextContent();
    pages.push({ page: index, text: content.items.filter(item => 'str' in item).map(item => item.str + (item.hasEOL ? '\n' : ' ')).join('') });
    page.cleanup();
  }
  await fs.writeFile(path.join(outRoot, name + '.json'), JSON.stringify(pages, null, 2));
  await fs.writeFile(path.join(outRoot, name + '.txt'), pages.map(p => `\n=== PDF PAGE ${p.page} ===\n${p.text}\n`).join(''));
  console.log(JSON.stringify({ name, pages: pages.length, characters: pages.reduce((n, p) => n + p.text.length, 0) }));
  await pdf.destroy();
}
