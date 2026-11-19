#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="el9_gpu|public_a100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --job-name=htd_b3_qeq
#SBATCH --output=/home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/slurm-qeq-%j.out
set -e
PY=/home/moriya2524/.conda/envs/newtonnet/bin/python
export MLPUI_NEIGHBORS=python OMP_NUM_THREADS=8 PYTHONPATH=/scratch/moriya2524/mlpui_qeq/src
export FAIRCHEM_CACHE_DIR=/scratch/moriya2524/fairchem_cache HF_HUB_OFFLINE=1
cd /scratch/moriya2524/mlpui_qeq/src
echo "commit $(git rev-parse HEAD)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date
$PY /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/verify_manifest.py /scratch/moriya2524/qeq_stage1_b3lyp_qc63_v1
$PY -m scripts.train /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/train_qeq_stage1.yaml > /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/history_qeq_s42.json
date
$PY /home/moriya2524/mlp/runs/cm_qeq_v1/eval_qeq.py /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/qeq_stage1_s42/best.pt /scratch/moriya2524/qeq_stage1_b3lyp_qc63_v1 /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/eval_qeq_s42.json
for role in train validation test guard; do
  $PY -m scripts.qeq_residual /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/qeq_stage1_s42/best.pt /scratch/moriya2524/qeq_stage1_b3lyp_qc63_v1/$role /scratch/moriya2524/htdlh_qeq_residual_b3lyp_qc63_s42/$role --batch-size 64 --device cuda --dtype float64 > /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/residual_$role.json
done
date
$PY /home/moriya2524/mlp/runs/htdlh_qeq_b3lyp_v1/uma_residual_b3lyp.py
date; echo ALLDONE
