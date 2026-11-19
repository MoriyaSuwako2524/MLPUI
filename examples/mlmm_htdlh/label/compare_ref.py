import re, sys, numpy as np
def rd(p):
    t=open(p).read()
    mol=re.search(r"\$molecule\n(.*?)\$end",t,re.S).group(1).strip().splitlines()
    q=np.array([[float(x) for x in l.split()[1:4]] for l in mol[1:]]); el=[l.split()[0] for l in mol[1:]]
    ext=np.array([[float(x) for x in l.split()] for l in re.search(r"\$external_charges\n(.*?)\$end",t,re.S).group(1).strip().splitlines()])
    return mol[0],el,q,ext
a=rd(sys.argv[1]); b=rd(sys.argv[2])
print("charge/mult", a[0], "|", b[0]); print("nQM", len(a[1]), len(b[1]), "elements equal", a[1]==b[1])
print("QM max |dx| (A)", np.abs(a[2]-b[2]).max() if a[2].shape==b[2].shape else "shape mismatch")
print("nMM", len(a[3]), len(b[3]), "sum q", a[3][:,3].sum().round(5), b[3][:,3].sum().round(5))
if a[3].shape==b[3].shape:
    print("MM max |dx|", np.abs(a[3][:,:3]-b[3][:,:3]).max(), "max |dq|", np.abs(a[3][:,3]-b[3][:,3]).max())
