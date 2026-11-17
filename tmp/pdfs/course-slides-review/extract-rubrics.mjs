import fs from 'node:fs/promises';
import path from 'node:path';
import { getDocument } from 'file:///C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdfjs-dist/legacy/build/pdf.mjs';
const folder = 'D:/Desktop/uni_of_edinburgh/year4/machine_learning_systems/cw';
const output = path.join(folder, 'edin-mls-26-autumn/tmp/pdfs/rubric-review');
await fs.mkdir(output, { recursive: true });
for (const file of ['01_submission_requirements', '02_group_report_rubric', '03_individual_review_requirements', '04_individual_review_rubric', '05_video_production_guide', 'week4coursework-combined']) {
  const data = new Uint8Array(await fs.readFile(path.join(folder, file + '.pdf')));
  const pdf = await getDocument({ data, useSystemFonts: true, disableFontFace: true }).promise;
  const pages = [];
  for (let page = 1; page <= pdf.numPages; page++) {
    const content = await (await pdf.getPage(page)).getTextContent();
    const text = content.items.filter(item => 'str' in item).map(item => item.str + (item.hasEOL ? '\n' : ' ')).join('');
    pages.push({ page, text });
  }
  await fs.writeFile(path.join(output, file + '.json'), JSON.stringify(pages, null, 2));
  console.log(JSON.stringify({ file, pages: pdf.numPages, characters: pages.reduce((sum, p) => sum + p.text.length, 0) }));
  await pdf.destroy();
}
