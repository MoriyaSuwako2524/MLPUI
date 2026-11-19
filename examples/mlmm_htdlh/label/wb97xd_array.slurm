#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 2-12:00:00
#SBATCH --exclude=n122
#SBATCH --array=0-30
#SBATCH --job-name=htd_wb97xd
#SBATCH -o /scratch/moriya/htdlh_labels_v1/logs/htd_wb97xd_w%a-%A.out
module load intel/2021.2.0
export QC=/scratch/moriya/software/qchem_default QCAUX=/home/chance/software/qcaux
source $QC/bin/qchem.setup.sh
unset LD_PRELOAD
export QCSCRATCH=/tmp/$USER.$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID; mkdir -p $QCSCRATCH
trap 'rm -rf $QCSCRATCH' EXIT
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
R=/scratch/moriya/htdlh_labels_v1
PY=/scratch/moriya/python/my_env/qmhub-mlpui/bin/python
W=$(printf %02d $SLURM_ARRAY_TASK_ID)
nok=0; nbad=0
for f in $(seq -w 0 199); do
  d=$R/frames/$W/$f
  [ -s $d/embed.inp ] || { echo "MISSING_EMBED $W/$f"; nbad=$((nbad+1)); continue; }
  [ -s $d/b3lyp_qmmm/qchem.inp ] || $PY $R/tools/build_labels.py $d/embed.inp $d > /dev/null
  for k in wb97xd_qmmm wb97xd_gas; do
    cd $d/$k
    if [ -f qchem.out ] && grep -q "Thank you very much" qchem.out; then nok=$((nok+1)); continue; fi
    t0=$(date +%s)
    qchem -nt 16 qchem.inp qchem.out save > run.log 2>&1
    rm -rf $QCSCRATCH/save qchem.out.esp qchem.out.efld
    if grep -q "Thank you very much" qchem.out; then nok=$((nok+1)); s=ok; else nbad=$((nbad+1)); s=FAIL; fi
    echo "$W/$f/$k $(( $(date +%s)-t0 ))s $s"
  done
done
echo "htd_wb97xd w$W DONE ok=$nok bad=$nbad $(date)"
[ $nbad -eq 0 ]
