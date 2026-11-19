import re, sys, glob, numpy as np
from scipy.io import netcdf_file
nc=netcdf_file(sys.argv[2],'r',mmap=False); X=nc.variables['coordinates'][:].astype(float)
def xi_from(P,a,b,c): return np.linalg.norm(P[...,a,:]-P[...,b,:],axis=-1)-np.linalg.norm(P[...,b,:]-P[...,c,:],axis=-1)
xin=xi_from(X,1913,1914,3135)
dumps=sorted(glob.glob(sys.argv[1]+'/embed_*.inp')); out=[]
for d in dumps:
    t=open(d).read(); mol=re.search(r"\$molecule\n(.*?)\$end",t,re.S).group(1).strip().splitlines()[1:]
    Q=np.array([[float(x) for x in l.split()[1:4]] for l in mol])
    xd=xi_from(Q,3,4,14); j=int(np.argmin(abs(xin-xd))); out.append((d.split('_')[-1][:4],round(xd,5),j,round(abs(xin[j]-xd),6)))
print('n_dumps',len(dumps),'n_frames',len(X))
for o in out[:4]+out[-3:]: print(o)
js=[o[2] for o in out]; print('unique frames matched',len(set(js)),'max |dxi|',max(o[3] for o in out))
