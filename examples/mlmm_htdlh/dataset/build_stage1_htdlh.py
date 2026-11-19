import numpy as np, re, os, sys, json, hashlib, time
from concurrent.futures import ProcessPoolExecutor
M = sys.argv[1]                      # b3lyp | wb97xd
R = "/scratch/moriya/htdlh_labels_v1/frames"
OUT = f"/scratch/moriya/htdlh_labels_v1/datasets/qeq_stage1_{M}_qc63_v1"
LEVEL = {"b3lyp": "B3LYP-D3(BJ)/6-31G*, SG-1 (xc_grid 1), thresh 12, scf_convergence 9",
         "wb97xd": "wB97X-D/6-31+G*, xc_grid 3, thresh 12, scf_convergence 8"}[M]
HA2EV = 27.211386245988; B2A = 0.529177210903
EPAT = re.compile(r"Total energy (?:in the final basis set )?=\s+(-?\d+\.\d+)")
STAGES = ["05ps", "10ps", "15ps", "20ps"]
def energy(t):
    assert "Thank you very much" in t[-3000:]; return float(EPAT.findall(t)[-1])
def version(t):
    m = re.search(r"Q-Chem (\d+\.\d+)", t); return m.group(1) if m else "?"
def gblock(t):
    i = t.find(" Gradient of SCF Energy\n"); assert i >= 0
    cols = {}; cur = None
    for l in t[i:].splitlines()[1:]:
        s = l.split()
        if not s: continue
        if "Max gradient" in l: break
        if all(x.isdigit() for x in s): cur = [int(x) for x in s]; continue
        k = int(s[0])
        for a, x in zip(cur, s[1:]): cols.setdefault(a, [0., 0., 0.])[k - 1] = float(x)
    return np.array([cols[a] for a in sorted(cols)])
def plotesp(f):
    rows = [l.split() for l in open(f).read().splitlines()[4:] if l.strip()]
    return np.array([[float(x) for x in r] for r in rows])
def mk(t):
    i = t.rfind("Merz-Kollman ESP Net Atomic Charges"); assert i >= 0; L = t[i:].splitlines()[4:]
    q = []
    for l in L:
        s = l.split()
        if len(s) >= 3 and s[0].isdigit(): q.append(float(s[2]))
        elif q: break
    return np.array(q)
def parse(wf):
    w, f = wf; d = f"{R}/{w:02d}/{f:03d}"
    try:
        inp = open(d + f"/{M}_qmmm/qchem.inp").read()
        mol = re.search(r"\$molecule\s*\n(.*?)\$end", inp, re.S).group(1).strip().split("\n")
        chg = int(mol[0].split()[0]); z = np.array([int(l.split()[0]) for l in mol[1:]]); X = np.array([[float(v) for v in l.split()[1:4]] for l in mol[1:]])
        ext = np.array([[float(v) for v in l.split()] for l in re.search(r"\$external_charges\s*\n(.*?)\$end", inp, re.S).group(1).strip().split("\n")])
        n = len(ext); nq = len(z)
        tq = open(d + f"/{M}_qmmm/qchem.out").read(); Eq = energy(tq); mkq_qmmm = mk(tq); assert mkq_qmmm.shape == (nq,)
        ef = np.loadtxt(d + f"/{M}_qmmm/efield.dat"); assert ef.shape == (n + nq, 3)
        fq, gq = ef[:n], ef[n:]
        pe = plotesp(d + f"/{M}_qmmm/plot.esp"); assert pe.shape == (n, 4) and np.abs(pe[:, :3] * B2A - ext[:, :3]).max() < 1e-5
        tg = open(d + f"/{M}_gas/qchem.out").read(); Eg = energy(tg); gg = gblock(tg); assert gg.shape == (nq, 3)
        mkq = mk(tg); assert mkq.shape == (nq,)
        ge = plotesp(d + f"/{M}_gas/plot.esp"); assert ge.shape == (n, 4) and np.abs(ge[:, :3] * B2A - ext[:, :3]).max() < 1e-5
        fe = np.loadtxt(d + f"/{M}_gas/efield.dat"); assert fe.shape == (n, 3)
        src = dict(kv.split("=") for kv in open(d + "/source").read().split())
        newton = np.abs(gq.sum(0) - (ext[:, 3:4] * fq).sum(0)).max()
        dmin = np.sqrt(((ext[:, None, :3] - X[None]) ** 2).sum(-1)).min()
        xi = np.linalg.norm(X[3] - X[4]) - np.linalg.norm(X[4] - X[14])
        return dict(w=w, f=f, z=z, pos=X, charge=chg, mm_pos=ext[:, :3], mm_charges=ext[:, 3],
                    e_qmmm=Eq * HA2EV, f_qmmm=-gq * HA2EV / B2A, e_gas=Eg * HA2EV, f_gas=-gg * HA2EV / B2A,
                    phi=pe[:, 3] * HA2EV, gphi=-fq * HA2EV / B2A, gas_phi=ge[:, 3] * HA2EV, gas_gphi=-fe * HA2EV / B2A,
                    mk=mkq, mk_qmmm=mkq_qmmm, newton=newton, gsum=np.abs(gg.sum(0)).max(), dmin=dmin, xi=xi,
                    stage=STAGES.index(src["stage"]), ncf=int(src["nc_frame"]), ver=(version(tq), version(tg)))
    except Exception as e:
        return (w, f, repr(e)[:200])
def split_of(f):
    return "train" if f < 140 else "guard" if f < 150 else "validation" if f < 170 else "guard" if f < 180 else "test"
if __name__ == "__main__":
    t0 = time.time(); todo = [(w, f) for w in range(31) for f in range(200)]
    with ProcessPoolExecutor(int(os.environ.get("NPROC", "16"))) as ex: recs = list(ex.map(parse, todo, chunksize=25))
    bad = [r for r in recs if isinstance(r, tuple)]; good = [r for r in recs if isinstance(r, dict)]
    print("parsed", len(good), "failed", len(bad), bad[:5], "%.0fs" % (time.time() - t0), flush=True)
    assert not bad
    assert len({tuple(r["z"]) for r in good}) == 1 and {r["ver"] for r in good} == {("6.3", "6.3")}
    assert all(r["stage"] * 50 + r["ncf"] == r["f"] for r in good)
    dE = np.array([r["e_qmmm"] - r["e_gas"] for r in good]); dF = np.array([np.abs(r["f_qmmm"] - r["f_gas"]).max() for r in good])
    Fq = np.array([np.abs(r["f_qmmm"]).max() for r in good])
    def robust_z(x): med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826; return (x - med) / mad
    zE = robust_z(dE)
    chk = dict(n_frames=len(good), qchem_version="6.3 (devel) for every qmmm and gas output",
               max_newton_au=float(max(r["newton"] for r in good)), max_gas_grad_sum_au=float(max(r["gsum"] for r in good)),
               min_qm_mm_dist_A=float(min(r["dmin"] for r in good)), n_mm_sites_min_max=[int(min(len(r["mm_charges"]) for r in good)), int(max(len(r["mm_charges"]) for r in good))],
               dE_eV_mean_std_min_max=[float(dE.mean()), float(dE.std()), float(dE.min()), float(dE.max())],
               max_abs_dF_component_eVA=float(dF.max()), max_abs_F_qmmm_component_eVA=float(Fq.max()),
               n_dE_robust_z_gt6=int((np.abs(zE) > 6).sum()),
               xi_min_max=[float(min(r["xi"] for r in good)), float(max(r["xi"] for r in good))])
    print(json.dumps(chk), flush=True)
    man = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), system="htdlh PETase-like hydrolase, Cys121->His202 proton transfer",
               source=R + "/<ww>/<fff>/{embed.inp,source," + M + "_qmmm," + M + "_gas}",
               level=LEVEL + "; Q-Chem 6.3 (devel) /scratch/moriya/software/qchem_default; QM/MM embedding = QMHub lrec 10 A + pbc replayed from alberto's umbrella trajectories (sander26 imin=5)",
               qm_region="(:121,202,170&!@N,HN,CA,HA,C,O)|(:PET), 57 atoms, charge -1; 0-based atoms 54,55,56 are link H (Amber lnk)",
               definition={"energy": "E_QMMM - E_gas (eV)", "forces": "F_QMMM - F_gas (eV/A)",
                           "mm_potential": "QM/MM ESP at embedding sites (eV/e), qmmm/plot.esp",
                           "mm_potential_gradient": "-field from qmmm/efield.dat first nMM rows (eV/(e A))",
                           "gas_mm_potential": "gas-density ESP at the same sites (gas/plot.esp, ESPGrid in bohr)",
                           "gas_mm_potential_gradient": "-field from gas/efield.dat",
                           "extra": "gas_energy, gas_forces, qmmm_energy, qmmm_forces, gas_mk_charges, qmmm_mk_charges, window, frame_in_window, stage (0..3 = 05ps..20ps), xi = r(SG-H)-r(H-ND1) (A, 0-based 3,4,14), min_qm_mm_dist"},
               filter="none", checks=chk,
               split="per window (200 frames, 100 fs apart, time-ordered): 0-139 train, 140-149 guard, 150-169 validation, 170-179 guard, 180-199 test",
               files={}, counts={})
    for sp in ("train", "validation", "test", "guard"):
        sel = [r for r in good if split_of(r["f"]) == sp]; od = f"{OUT}/{sp}"; os.makedirs(od + "/extra", exist_ok=True)
        off = np.concatenate([[0], np.cumsum([len(r["mm_charges"]) for r in sel])]).astype(np.int64)
        A = {"z": sel[0]["z"].astype(np.int64), "pos": np.stack([r["pos"] for r in sel]), "charge": np.array([float(r["charge"]) for r in sel]),
             "energy": np.array([r["e_qmmm"] - r["e_gas"] for r in sel]), "forces": np.stack([r["f_qmmm"] - r["f_gas"] for r in sel]),
             "mm_pos": np.concatenate([r["mm_pos"] for r in sel]), "mm_charges": np.concatenate([r["mm_charges"] for r in sel]), "mm_offsets": off,
             "mm_potential": np.concatenate([r["phi"] for r in sel]), "mm_potential_gradient": np.concatenate([r["gphi"] for r in sel]),
             "gas_mm_potential": np.concatenate([r["gas_phi"] for r in sel]), "gas_mm_potential_gradient": np.concatenate([r["gas_gphi"] for r in sel]),
             "frame_index": np.arange(len(sel), dtype=np.int64), "source_index": np.array([r["w"] * 200 + r["f"] for r in sel], np.int64),
             "short_contact": np.zeros(len(sel), np.int8),
             "extra/gas_energy": np.array([r["e_gas"] for r in sel]), "extra/gas_forces": np.stack([r["f_gas"] for r in sel]),
             "extra/qmmm_energy": np.array([r["e_qmmm"] for r in sel]), "extra/qmmm_forces": np.stack([r["f_qmmm"] for r in sel]),
             "extra/gas_mk_charges": np.stack([r["mk"] for r in sel]), "extra/qmmm_mk_charges": np.stack([r["mk_qmmm"] for r in sel]),
             "extra/window": np.array([r["w"] for r in sel], np.int64), "extra/frame_in_window": np.array([r["f"] for r in sel], np.int64),
             "extra/stage": np.array([r["stage"] for r in sel], np.int64), "extra/xi": np.array([r["xi"] for r in sel]),
             "extra/min_qm_mm_dist": np.array([r["dmin"] for r in sel])}
        for k, v in A.items():
            p_ = f"{od}/{k}.npy"; np.save(p_, v); man["files"][f"{sp}/{k}.npy"] = hashlib.sha256(open(p_, "rb").read()).hexdigest()
        man["counts"][sp] = len(sel)
        print(sp, len(sel), "mm points", int(off[-1]), "E std %.4f eV, F rms %.4f, (phi-gas) rms %.4f" % (A["energy"].std(), np.sqrt((A["forces"]**2).mean()), np.sqrt(((A["mm_potential"]-A["gas_mm_potential"])**2).mean())), flush=True)
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
    # small per-frame QC table for local plotting
    qc = np.array([[r["w"], r["f"], r["stage"], r["xi"], r["e_qmmm"], r["e_gas"], np.abs(r["f_qmmm"]).max(), np.abs(r["f_qmmm"] - r["f_gas"]).max(), r["newton"], r["dmin"], len(r["mm_charges"])] for r in good])
    np.savetxt("qc_per_frame.csv", qc, delimiter=",", header="window,frame_in_window,stage,xi_A,e_qmmm_eV,e_gas_eV,max_abs_F_qmmm_eVA,max_abs_dF_eVA,newton_au,min_qm_mm_dist_A,n_mm", comments="")
    json.dump(man, open("manifest.json", "w"), indent=1)
    print("BUILD_DONE", flush=True)
