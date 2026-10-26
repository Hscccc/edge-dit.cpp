import argparse, concurrent.futures, hashlib, json, subprocess, time
from pathlib import Path

parser=argparse.ArgumentParser(description="Download pinned Local Dream FLUX.2-klein assets and verify SHA256 on the device")
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[3]/'models')
parser.add_argument('--serial',required=True)
parser.add_argument('--transport-origin',default='https://huggingface.co')
parser.add_argument('--direct',action='store_true',help='Bypass proxy for the transport host')
args=parser.parse_args()
curl_network=['--noproxy','*'] if args.direct else []
assets = args.output / 'flux2-klein-4b'
assets.mkdir(parents=True, exist_ok=True)
entries = [
 ('zhiyuanasad/flux2_klein_adreno','233d195563e9f3edeb7c85c6f2b46cb9e7315411','diffusion_model.gguf',2256889280,'7d6fa5759f5a7443c0cc3dbba2bcf0978803beee9b1a967a091a50b6685f4936'),
 ('zhiyuanasad/flux2_klein_adreno','233d195563e9f3edeb7c85c6f2b46cb9e7315411','llm.gguf',2262670048,'d16827df1afc5231db8811bdc8b6ca7f4f845600f6ebca55ec0f6ae53ea8b481'),
 ('zhiyuanasad/flux2_klein_adreno','233d195563e9f3edeb7c85c6f2b46cb9e7315411','vae.safetensors',336213556,'d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5'),
 ('black-forest-labs/FLUX.2-klein-4b-fp8','5b4408e59397a4a37ccb46afe426d8ed86379441','flux-2-klein-4b-fp8.safetensors',4070624520,'97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6'),
]
serial=args.serial
remote='/data/local/tmp/edge-dit-npu/models/flux2-klein-4b'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def acquire(e):
 repo,rev,name,size,digest=e
 dst=assets/name
 if not (dst.exists() and dst.stat().st_size==size and sha(dst)==digest):
  part=Path(str(dst)+'.part')
  url=f'{args.transport_origin.rstrip("/")}/{repo}/resolve/{rev}/{name}?download=true'
  # Resolve once: range workers must not consume Hub resolver quota.
  url=subprocess.check_output(['curl',*curl_network,'--http1.1','-I','-L','--fail','--silent','--show-error','--retry','5','--retry-delay','15','--connect-timeout','20','--max-time','120',url,'-o','/dev/null','-w','%{url_effective}'],text=True)
  chunk_size=32<<20
  chunk_dir=assets/(name+'.chunks');chunk_dir.mkdir(exist_ok=True)
  prefix=(part.stat().st_size//chunk_size)*chunk_size if part.exists() else 0
  if part.exists() and part.stat().st_size==size:
   if sha(part)==digest:part.rename(dst);prefix=size
   else:prefix=0
  if not dst.exists() and part.exists():
   with part.open('r+b') as f:f.truncate(prefix)
  for old in chunk_dir.iterdir():
   if old.name.isdigit() and (int(old.name)+1)*chunk_size<=prefix:old.unlink()
  def chunk(index):
   start=index*chunk_size;end=min(size,start+chunk_size)-1
   if start<prefix or dst.exists():return None
   path=chunk_dir/str(index)
   if path.exists() and path.stat().st_size==end-start+1:return path
   for attempt in range(8):
    tmp=Path(str(path)+'.tmp')
    res=subprocess.run(['curl',*curl_network,'--http1.1','-L','--fail','--connect-timeout','15','--max-time','180','--silent','--show-error','-r',f'{start}-{end}',url,'-o',str(tmp)])
    if res.returncode==0 and tmp.exists() and tmp.stat().st_size==end-start+1:
     tmp.rename(path);return path
    time.sleep(min(60,10*(attempt+1)))
   raise RuntimeError('chunk failed: '+name+' '+str(index))
  count=(size+chunk_size-1)//chunk_size
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as chunks:
   for i,_ in enumerate(chunks.map(chunk,range(count))):
    if i%8==0:print('CHUNKS',name,i+1,count,flush=True)
  if not dst.exists():
   with part.open('ab') as f:
    for i in range(prefix//chunk_size,count):
     block_path=chunk_dir/str(i)
     with block_path.open('rb') as c:
      for block in iter(lambda:c.read(8<<20),b''):f.write(block)
     f.flush()
     block_path.unlink()
   if part.stat().st_size!=size or sha(part)!=digest: raise RuntimeError('hash mismatch: '+name)
   part.rename(dst)
 print('VERIFIED',name,flush=True)
 return dict(repo=repo,revision=rev,path=name,bytes=size,sha256=digest,local=str(dst),remote=f'{remote}/{name}')
subprocess.run(['adb','-s',serial,'shell',f'mkdir -p {remote}'],check=True)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for f in concurrent.futures.as_completed([pool.submit(acquire,e) for e in entries]):
  r=f.result(); results.append(r)
  (assets/'manifest.json').write_text(json.dumps(dict(schema_version=1,device_serial=serial,files=results),indent=2)+'\n')
  print('PUSH',r['path'],flush=True)
  subprocess.run(['adb','-s',serial,'push',r['local'],r['remote']+'.part'],check=True)
  actual=subprocess.check_output(['adb','-s',serial,'shell',f"sha256sum {r['remote']}.part"],text=True).split()[0]
  if actual!=r['sha256']: raise RuntimeError('device hash mismatch: '+r['path'])
  subprocess.run(['adb','-s',serial,'shell',f"mv {r['remote']}.part {r['remote']}"],check=True)
  r['device_verified']=True
  (assets/'manifest.json').write_text(json.dumps(dict(schema_version=1,device_serial=serial,files=results),indent=2)+'\n')
  print('DEVICE_VERIFIED',r['path'],flush=True)
subprocess.run(['adb','-s',serial,'push',str(assets/'manifest.json'),remote+'/manifest.json'],check=True)
print('ALL_MODELS_DEPLOYED',flush=True)
