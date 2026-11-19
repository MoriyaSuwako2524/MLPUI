# Worked example: htdlh ML/MM (B3LYP-D3(BJ)/6-31G*)

Tested files from the production run that took Amber QM/MM umbrella
trajectories (`<w>_pmf.nc`, 31 windows) of the htdlh Cys121->His202 proton
transfer to a UMA + QEq + TensorNet-QMMM model and new ML/MM umbrella
trajectories. The procedure, the reasons for each setting and the checks are in
[docs/mlmm-workflow.md](../../docs/mlmm-workflow.md). Files are copied verbatim:
they run on Pete (labels, MD) and Sooner (training) with the paths written in
them, and are a template, not a portable package.

| Folder | Step | Files |
|---|---|---|
| `label/` | 1. replay `.nc` -> QMHub embeddings | `replay.mdin`, `qmhub_dump.ini`, `fakeqc/qchem`, `fake_qchem.py`, `replay_array.slurm`, `map_dumps.py`, `compare_ref.py` |
| `label/` | 2. Q-Chem labels | `build_labels.py`, `b3lyp_array.slurm`, `wb97xd_array.slurm`, `validate_labels.py` |
| `dataset/` | 3. NpyDataset + manifest | `build_stage1_htdlh.py` |
| `train/` | 4. QEq -> QEq residual -> UMA residual -> dMLP | `submit_all.sh`, `run_qeq.sbatch`, `run_dmlp.sbatch`, `train_qeq_stage1.yaml`, `train_tnqmmm_uma_residual.yaml`, `uma_residual_b3lyp.py`, `verify_manifest.py`, `eval_qeq.py`, `eval_family.py` |
| `bundle/` | 5. model bundle | `htdlh_full_spec.json`, `htdlh_full_spec_fp64.json`, `uma_residual_manifest.json`, `README.txt`, `validate_bundle.py` |
| `md/` | 6. ML/MM MD | `mlmd.mdin`, `qmhub_mlp.ini`, `cv.rst`, `run.slurm`, `qmhub_wrapper.sh` |

Checkpoints and UMA weights are not included (see `bundle/README.txt` for
their sources and the UMA sha256).

## System-specific values to change

* QM region `qmmask="(:121,202,170&!@N,HN,CA,HA,C,O)|(:PET)"`, `qmcharge=-1`
  (57 QM atoms incl. 3 link H, 0-based 54-56) in `replay.mdin`, `mlmd.mdin`.
* Reaction coordinate xi = r(SG-H) - r(H-ND1): Amber atoms 1913/1914/3135
  (0-based, `.nc`) in `map_dumps.py`; QM indices 3/4/14 in `map_dumps.py` and
  `build_stage1_htdlh.py`; restraints in `cv.rst` (1-based Amber numbering).
* Frames: 4 stages (05ps..20ps) x 50 frames per window; split
  0-139 / 140-149 / 150-169 / 170-179 / 180-199 =
  train / guard / validation / guard / test (`build_stage1_htdlh.py`).
* `charge: -1` in the spec JSON and in the UMA residual script.
* Paths: `/projects/ok001/htdlh_protonTransfer` (source MD), `/scratch/moriya/...`,
  `/home/moriya/...` (Pete), `/scratch/moriya2524/...`, `/home/moriya2524/...`
  (Sooner), Python environments, `/scratch/van/Programs/sander26`.

## Results of this run (for reference)

* Replay: 50 dumps per stage = 50 frames, |dxi| <= 1e-6 A; ~10 s/frame.
* Labels: gas/QM-MM `plot.esp` sites agree to 1e-6 bohr; dataset max Newton
  error 9.9e-9 au; 4340 / 620 / 620 / 620 train / validation / test / guard frames;
  1434-1596 embedding sites per frame.
* Bundle (test frames, fp32 = fp64): RMSE E 0.405 kcal/mol, F 0.666 kcal/mol/A,
  ESP 0.402 kcal/mol/e, GESP 0.127.
