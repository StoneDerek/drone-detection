// Local preview with the same response headers used by Cloudflare Pages.
const http=require('node:http');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'../dist/cloudflare');
const headers={};
for(const line of fs.readFileSync(path.join(root,'_headers'),'utf8').split('\n')) {
  const match=line.match(/^\s+([^:]+):\s*(.*)$/);
  if(match)headers[match[1]]=match[2];
}
const mime={'.html':'text/html','.js':'text/javascript','.json':'application/json','.txt':'text/plain',
  '.md':'text/plain','.wav':'audio/wav','.png':'image/png','.bin':'application/octet-stream'};
http.createServer((request,response)=>{
  const pathname=decodeURIComponent(new URL(request.url,'http://localhost').pathname);
  const file=path.resolve(root,'.'+(pathname==='/'?'/index.html':pathname));
  if(!file.startsWith(root+'/')||!fs.existsSync(file)||!fs.statSync(file).isFile()) {
    response.writeHead(404,headers);response.end('Not found');return;
  }
  response.writeHead(200,{...headers,'Content-Type':mime[path.extname(file)]||'application/octet-stream'});
  fs.createReadStream(file).pipe(response);
}).listen(8012,'127.0.0.1',()=>console.log('Browser lab preview at http://127.0.0.1:8012'));
