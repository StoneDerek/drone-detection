import {build} from 'esbuild';
import {cp, mkdir, rm} from 'node:fs/promises';
await rm('../dist/cloudflare', {recursive:true, force:true});
await mkdir('../dist/cloudflare', {recursive:true});
await cp('public','../dist/cloudflare',{recursive:true});
await build({entryPoints:['src/app.js'], bundle:true, minify:true, format:'esm',
  outfile:'../dist/cloudflare/app.js', sourcemap:false, legalComments:'external'});
