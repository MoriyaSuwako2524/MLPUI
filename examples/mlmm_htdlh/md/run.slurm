#!/bin/bash
#SBATCH -p batch
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --exclude=n122
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --job-name=htm_w00
module load intel/2021.2.0
module load impi/2021.2.0
source /scratch/van/Programs/sander26/amber.sh
export MKLROOT=/opt/intel/oneapi/mkl/2021.2.0
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_sequential.so
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export I_MPI_THREAD_YIELD=3 I_MPI_THREAD_SLEEP=200
export PATH=/home/moriya/mlp/bin:$PATH QMHUB_THREADS=8 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
cd $SLURM_SUBMIT_DIR
hostname; date
srun -n 4 --cpu-bind=none sander.MPI -O -i mlmd.mdin -o mlmd.mdout -p step3_pbcsetup.parm7 -c start.rst7 -r mlmd.ncrst -x mlmd.nc -inf mlmd.mdinfo &
SPID=$!
sleep 300
while kill -0 $SPID 2>/dev/null; do
  if ! pgrep -u $USER -f "qmhub qmhub_mlp.ini" >/dev/null; then echo "WATCHDOG: qmhub not running, stopping sander"; kill $SPID; sleep 30; break; fi
  sleep 60
done
wait $SPID; echo "SANDER_EXIT $?"
date
