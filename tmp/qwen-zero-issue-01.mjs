const task = await taskSpace('Publish Qwen zero-score investigation');
const page = task.page('p1');
await page.goto('https://github.com/JackSuuu/edin-mls-26-autumn/issues');
console.log({ taskSpaceId: task.spaceId, page: page.label });
console.log(await page.snapshot({ scope: 'full_page' }));
