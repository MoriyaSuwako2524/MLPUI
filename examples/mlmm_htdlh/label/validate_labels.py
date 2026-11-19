import numpy as np, re, sys
B=sys.argv[1]
t=open(B+'/embed.inp').read()
ext=np.array([[float(x) for x in l.split()] for l in re.search(r"\$external_charges\n(.*?)\$end",t,re.S).group(1).strip().splitlines()])
nqm=len(re.search(r"\$molecule\n(.*?)\$end",t,re.S).group(1).strip().splitlines())-1; nmm=len(ext)
A2B=1/0.529177210903
def esp(p): return np.loadtxt(p, skiprows=4)
print('nQM',nqm,'nMM',nmm)
for m in ['b3lyp','wb97xd']:
    q=esp(f'{B}/{m}_qmmm/plot.esp'); g=esp(f'{B}/{m}_gas/plot.esp')
    eq=np.loadtxt(f'{B}/{m}_qmmm/efield.dat'); eg=np.loadtxt(f'{B}/{m}_gas/efield.dat')
    print(m,'plot.esp qmmm/gas',q.shape,g.shape,'| site diff gas-qmmm (bohr)',np.abs(q[:,:3]-g[:,:3]).max(),'| qmmm sites-ext (bohr)',np.abs(q[:,:3]-ext[:,:3]*A2B).max(),'| efield rows',eq.shape,eg.shape)
    gq=eq[nmm:]; fmm=ext[:,3:4]*eq[:nmm]
    print('  sum grad_QM',gq.sum(0).round(6),' sum F_MM',fmm.sum(0).round(6),' |grad_QM - F_MM| sum',np.abs(gq.sum(0)-fmm.sum(0)).max())
    print('  ESP RMS qmmm-gas (au)',np.sqrt(((q[:,3]-g[:,3])**2).mean()).round(5),' field RMS qmmm-gas',np.sqrt(((eq[:nmm]-eg[:nmm])**2).mean()).round(5))
    # gas gradient block
    o=open(f'{B}/{m}_gas/qchem.out').read(); print('  gas has Gradient block:', 'Gradient of SCF Energy' in o, '| qmmm has:', 'Gradient of SCF Energy' in open(f'{B}/{m}_qmmm/qchem.out').read())
