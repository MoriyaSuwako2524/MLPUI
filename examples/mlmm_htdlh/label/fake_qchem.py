import os, sys, re, shutil, numpy as np
args=[a for a in sys.argv[1:] if not a.startswith('-') and not a.isdigit()]
inp=args[0]; txt=open(inp).read()
mol=re.search(r"\$molecule\n(.*?)\$end",txt,re.S).group(1).strip().splitlines()[1:]
ext=re.search(r"\$external_charges\n(.*?)\$end",txt,re.S).group(1).strip().splitlines()
nqm,nmm=len(mol),len(ext)
dd=os.environ['DUMPDIR']; os.makedirs(dd,exist_ok=True)
prev=sorted(f for f in os.listdir(dd) if f.endswith('.inp'))
same=False
if prev:
    pt=open(os.path.join(dd,prev[-1])).read()
    same=(pt==txt)
if not same:
    shutil.copyfile(inp, os.path.join(dd,f"embed_{len(prev):04d}.inp"))
os.makedirs('save',exist_ok=True)
np.zeros(2).tofile('save/99.0'); np.zeros(nqm*3).tofile('save/131.0')
np.zeros(nmm).tofile('save/1521.0'); np.zeros(nmm*3).tofile('save/329.0')
open(args[1] if len(args)>1 else 'qchem.out','w').write(f"fake nqm={nqm} nmm={nmm}\n")
