// Cloudflare Pages content-addressing, matching pinned Wrangler4.65.0.
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {hash}=require('blake3-wasm');
const [root,output]=process.argv.slice(2);
const files=[];
function visit(directory) {
  for(const entry of fs.readdirSync(directory,{withFileTypes:true})) {
    const absolute=path.join(directory,entry.name);
    if(entry.isDirectory())visit(absolute);
    else if(!['_headers','_redirects'].includes(entry.name)) {
      const data=fs.readFileSync(absolute),relative='/'+path.relative(root,absolute).split(path.sep).join('/');
      files.push({path:absolute,url:relative,size:data.length,
        hash:hash(data.toString('base64')+path.extname(absolute).substring(1)).toString('hex').slice(0,32),
        sha256:crypto.createHash('sha256').update(data).digest('hex')});
    }
  }
}
visit(root);
fs.writeFileSync(output,JSON.stringify(files,null,2)+'\n');
console.log(JSON.stringify({files:files.length,bytes:files.reduce((a,b)=>a+b.size,0)}));
