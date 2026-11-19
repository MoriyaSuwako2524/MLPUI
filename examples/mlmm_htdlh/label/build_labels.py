import os, re, sys
B2A=0.529177210903
RB3_QMMM="""$rem
jobtype force
method b3lyp
basis 6-31g*
dft_d D3_BJ
scf_convergence 9
thresh 12
xc_grid 1
max_scf_cycles 2000
mem_total 24000
mem_static 1000
esp_charges 1
qm_mm true
symmetry off
sym_ignore true
no_reorient true
print_input false
qmmm_print true
skip_charge_self_interact true
$end
"""
RB3_GAS="""$rem
jobtype force
method b3lyp
basis 6-31g*
dft_d D3_BJ
scf_convergence 9
thresh 12
xc_grid 1
max_scf_cycles 2000
mem_total 24000
mem_static 1000
esp_charges 1
symmetry false
sym_ignore true
no_reorient true
esp_grid {n}
esp_efield 1
$end
"""
RWB_QMMM="""$rem
JOBTYPE FORCE
METHOD wB97X-D
BASIS 6-31+G*
SYMMETRY FALSE
SYM_IGNORE TRUE
NO_REORIENT TRUE
SCF_CONVERGENCE 8
THRESH 12
MAX_SCF_CYCLES 2000
XC_GRID 3
MEM_TOTAL 24000
MEM_STATIC 1000
ESP_CHARGES 1
QM_MM TRUE
QMMM_PRINT TRUE
ESP_EFIELD 1
SKIP_CHARGE_SELF_INTERACT TRUE
$end
"""
RWB_GAS="""$rem
JOBTYPE FORCE
METHOD wB97X-D
BASIS 6-31+G*
SYMMETRY FALSE
SYM_IGNORE TRUE
NO_REORIENT TRUE
SCF_CONVERGENCE 8
THRESH 12
MAX_SCF_CYCLES 2000
XC_GRID 3
MEM_TOTAL 24000
MEM_STATIC 1000
ESP_CHARGES 1
ESP_GRID {n}
ESP_EFIELD 1
$end
"""
def build(embed, d):
    txt=open(embed).read()
    mol=re.search(r"\$molecule\n.*?\$end\n",txt,re.S).group(0)
    extb=re.search(r"\$external_charges\n.*?\$end\n",txt,re.S).group(0)
    ext=extb.strip().splitlines()[1:-1]
    grid="".join("%.12f %.12f %.12f\n"%tuple(float(x)/B2A for x in l.split()[:3]) for l in ext)
    for tag,rem,gas in [("b3lyp_qmmm",RB3_QMMM,False),("b3lyp_gas",RB3_GAS,True),("wb97xd_qmmm",RWB_QMMM,False),("wb97xd_gas",RWB_GAS,True)]:
        o=os.path.join(d,tag); os.makedirs(o,exist_ok=True)
        if gas:
            open(o+"/qchem.inp","w").write(rem.format(n=len(ext))+"\n"+mol)
            open(o+"/ESPGrid","w").write(grid)
        else:
            open(o+"/qchem.inp","w").write(rem+"\n"+mol+"\n"+extb)
    return len(ext)
if __name__=="__main__":
    print(build(sys.argv[1],sys.argv[2]))
