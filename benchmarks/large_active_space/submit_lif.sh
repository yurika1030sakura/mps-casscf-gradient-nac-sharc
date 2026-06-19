#!/bin/bash
#SBATCH -J lif_xing
#SBATCH -A ryl_lab
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH --array=0-11
#SBATCH -o slurm_lif_%A_%a.out
#SBATCH -e slurm_lif_%A_%a.err
set -euo pipefail

RVALS=(2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0)
R=${RVALS[$SLURM_ARRAY_TASK_ID]}

SCR=/n/netscratch/woo_lab/Lab/yulili/lif_xing/R${R}
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=/n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11
WORK=/n/home04/yulili/daisuan/dmrg_sacasscf_response_public/benchmarks/large_active_space
cd $WORK
echo "LiF R=${R} start $(date) host $(hostname)"
$PY run_lif_avoided_crossing.py --R $R --basis 6-31G --ncas 6 --nelecas 6 \
    --bond-dim 400 --threads 8 --stack-mem-mb 6000 \
    --out $WORK/data/lif/lif_R${R}.json
echo "done LiF R=${R} $(date)"
