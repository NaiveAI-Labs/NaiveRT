// Publish only browser assets. No runtime code, tests, or evidence are deployed.
import {mkdir, copyFile, rm} from 'node:fs/promises';
const root = new URL('./', import.meta.url);
const out = new URL('./dist/', root);
await rm(out, {recursive: true, force: true});
await mkdir(out, {recursive: true});
for (const name of ['index.html', 'studio.js', 'studio.css', 'core.js', 'markdown.js', 'vendor/markdown-it/markdown-it.mjs', 'vendor/markdown-it/LICENSE', 'curve.js', 'endpoint.js', 'queue.js', 'connection.js', 'logo.png']) {
  await mkdir(new URL('.', new URL(name, out)), {recursive: true});
  await copyFile(new URL(name, root), new URL(name, out));
}
console.log('Studio static site: dist/');
