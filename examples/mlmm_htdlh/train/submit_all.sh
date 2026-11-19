#!/bin/bash
# run after the dataset is copied to /scratch/moriya2524/qeq_stage1_b3lyp_qc63_v1
cd /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1
A=$(sbatch --parsable run_qeq.sbatch); B=$(sbatch --parsable --dependency=afterok:$A run_dmlp.sbatch)
echo "QEq job $A, dMLP job $B" | tee jobs.txt
