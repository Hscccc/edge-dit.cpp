#!/usr/bin/env python3
import argparse,datetime,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description="Run FLUX.2-klein on Android")
p.add_argument('--serial',required=True)
p.add_argument('--width',type=int,default=256)
p.add_argument('--height',type=int,default=256)
p.add_argument('--steps',type=int,default=4)
p.add_argument('--backend',choices=['hiai','cpu'],default='hiai')
p.add_argument('--threads',type=int,default=4)
p.add_argument('--prompt',default='a glass teapot on a wooden table')
p.add_argument('--output',type=Path)
a=p.parse_args()
repo=Path(__file__).resolve().parents[2]
out=a.output or repo/'benchmark/results/mobile'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
out.mkdir(parents=True,exist_ok=False)
remote='/data/local/tmp/edge-dit-npu'
model=remote+'/models/flux2-klein-4b'
cmd=[remote+'/ed-cli','--backend',a.backend,'--diffusion-model',model+'/diffusion_model.gguf',
     '--llm',model+'/llm.gguf','--vae',model+'/vae.safetensors','--prompt',a.prompt,
     '--width',str(a.width),'--height',str(a.height),'--steps',str(a.steps),'--guidance','1.0',
     '--cfg-scale','1','--seed','42','--threads',str(a.threads),'--vae-tiling','off','--output',remote+'/output.png']
(out/'run.json').write_text(json.dumps(dict(device_serial=a.serial,command=cmd),indent=2)+'\n')
with (out/'native.log').open('w') as log:
 result=subprocess.run(['adb','-s',a.serial,'shell',shlex.join(cmd)],stdout=log,stderr=subprocess.STDOUT)
if result.returncode==0:
 subprocess.run(['adb','-s',a.serial,'pull',remote+'/output.png',str(out/'output.png')],check=True)
print(out)
raise SystemExit(result.returncode)
