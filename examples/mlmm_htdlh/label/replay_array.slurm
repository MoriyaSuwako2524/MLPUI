#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH --exclude=n122
#SBATCH --array=0-30
#SBATCH --job-name=htd_replay
#SBATCH -o /scratch/moriya/htdlh_labels_v1/logs/replay_w%a-%A.out
module load intel/2021.2.0
module load impi/2021.2.0
source /scratch/van/Programs/sander26/amber.sh
export MKLROOT=/opt/intel/oneapi/mkl/2021.2.0
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_sequential.so
export PATH=/home/moriya/mlp/bin:$PATH QMHUB_THREADS=2 OMP_NUM_THREADS=2
R=/scratch/moriya/htdlh_labels_v1
SRC=/projects/ok001/htdlh_protonTransfer
PY=/scratch/moriya/python/my_env/qmhub-mlpui/bin/python
W=$(printf %02d $SLURM_ARRAY_TASK_ID)
si=0; fail=0
for S in 05ps 10ps 15ps 20ps; do
  NC=$SRC/results/$S/${W}_trj/${W}_pmf.nc
  n_have=0; for f in $(seq 0 49); do [ -s $R/frames/$W/$(printf %03d $((si*50+f)))/embed.inp ] && n_have=$((n_have+1)); done
  if [ $n_have -eq 50 ]; then echo "w$W $S already done"; si=$((si+1)); continue; fi
  P=$R/replay/$W/$S; rm -rf $P; mkdir -p $P/qmhub; cd $P
  ln -sf $SRC/input/step3_pbcsetup.parm7 .
  cp $R/tools/qmhub_dump.ini qmhub.ini
  sed "s#__BASEDIR__#$P/qmhub#" $R/tools/replay.mdin > replay.mdin
  export DUMPDIR=$P/dump; t0=$(date +%s)
  PATH=$R/tools/fakeqc:$PATH sander -O -i replay.mdin -o replay.mdout -p step3_pbcsetup.parm7 -c $SRC/input/reactantCorrected.rst7 -y $NC -r replay.rst7 -x replay.nc -inf replay.mdinfo || { echo "w$W $S SANDER_FAIL"; fail=1; si=$((si+1)); continue; }
  nd=$(ls $DUMPDIR | wc -l)
  chk=$($PY $R/tools/map_dumps.py $DUMPDIR $NC | grep 'unique frames matched')
  echo "w$W $S replay $(( $(date +%s)-t0 ))s ndump=$nd | $chk"
  um=$(echo $chk | awk '{print $4}'); mx=$(echo $chk | awk '{print $NF}')
  if [ "$nd" != 50 ] || [ "$um" != 50 ] || [ "$(awk -v m=$mx 'BEGIN{print (m>1e-4)?1:0}')" = 1 ]; then echo "w$W $S MAPPING_FAIL"; fail=1; si=$((si+1)); continue; fi
  for f in $(seq 0 49); do
    d=$R/frames/$W/$(printf %03d $((si*50+f))); mkdir -p $d
    cp $DUMPDIR/embed_$(printf %04d $f).inp $d/embed.inp
    echo "window=$W stage=$S nc=$NC nc_frame=$f frame_in_window=$((si*50+f))" > $d/source
  done
  rm -f $P/replay.nc $P/qmhub/qmmm.inp $P/qmhub/qmmm.out
  si=$((si+1))
done
echo "REPLAY w$W DONE fail=$fail $(date)"
exit $fail
