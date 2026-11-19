#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="el9_gpu|public_a100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --job-name=htd_b3_dmlp
#SBATCH --output=/home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/slurm-dmlp-%j.out
set -e
PY=/home/moriya2524/.conda/envs/newtonnet/bin/python
export MLPUI_NEIGHBORS=python OMP_NUM_THREADS=8 PYTHONPATH=/scratch/moriya2524/mlpui_qeq/src
cd /scratch/moriya2524/mlpui_qeq/src
echo "commit $(git rev-parse HEAD)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date
$PY -m scripts.train /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/train_tnqmmm_uma_residual.yaml > /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/history_dmlp_s42.json
date
$PY /home/moriya2524/mlp/runs/cm_qeq_v1/eval_family.py tensornet_qmmm /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/tnqmmm_uma_residual_s42/best.pt /scratch/moriya2524/htdlh_qeq_uma_residual_b3lyp_qc63_s42 /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/eval_dmlp_s42.json
date; echo ALLDONE
