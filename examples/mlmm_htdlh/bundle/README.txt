htdlh B3LYP QM/MM model bundle (UMA-s-1p2 + offset + QEq + TensorNet-QMMM dMLP), built on Sooner 2026-11-15.
System: htdlh PETase-like hydrolase, Cys121->His202 proton transfer; QM = (:121,202,170&!@N,HN,CA,HA,C,O)|(:PET),
57 atoms incl. 3 link H (0-based 54,55,56), charge -1, spin 1.
Trained on B3LYP-D3(BJ)/6-31G* labels (Q-Chem 6.3 devel, SG-1, thresh 12, scf_convergence 9) in the QMHub lrec 10 A + pbc embedding,
dataset qeq_stage1_b3lyp_qc63_v1 (4340/620/620/620 frames).
E = E_UMA(uma-s-1p2, omol, charge -1, spin 1) + energy_offset_eV + E_QEq + E_dMLP ; units A / eV / e.
Sources:
  models/uma-s-1p2.pt                 <- symlink, NOT included (sha256 ba5c0d912efa22dc238e5fb1b5b7f66ee2e68c48c1b95b7cfd7fe1da5938398b)
  models/qeq_best.pt                  <- /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/qeq_stage1_s42/best.pt (H64 L2 rc5)
  models/dmlp_best.pt                 <- /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/tnqmmm_uma_residual_s42_cont/best.pt
  models/uma_residual_manifest.json   <- /scratch/moriya2524/htdlh_qeq_uma_residual_b3lyp_qc63_s42/manifest.json (energy_offset_eV)
Specs: htdlh_full_spec.json (float32), htdlh_full_spec_fp64.json (float64), device cpu, paths relative to the spec.
Validation: validation/test8_reference.npz (see validate.log). Verify the copy: sha256sum -c SHA256SUMS
