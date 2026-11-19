#!/bin/bash
unset LD_PRELOAD
export OMP_NUM_THREADS=${QMHUB_THREADS:-4} MKL_NUM_THREADS=${QMHUB_THREADS:-4}
# mlpui.model_loader imports the repo-root 'scripts' package, so the MLPUI root must be on PYTHONPATH
export PYTHONPATH=/home/moriya/mlp/sources/MLPUI${PYTHONPATH:+:$PYTHONPATH}
# -P: do not put CWD on sys.path (sander's FIFO dir is often named 'qmhub' and would shadow the package)
exec /scratch/moriya/python/my_env/qmhub-mlpui/bin/python -P -m qmhub "$@"
