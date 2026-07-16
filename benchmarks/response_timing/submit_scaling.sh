#!/bin/bash
#SBATCH -J schur_scaling
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 18:00:00
#SBATCH --array=0-3
#SBATCH -o slurm_scaling_%A_%a.out
#SBATCH -e slurm_scaling_%A_%a.err
set -euo pipefail

SYS=(heh_cas22 h2o_cas44 n2_cas66 c2_cas88)
KEY=${SYS[$SLURM_ARRAY_TASK_ID]}

# block2 / pyscf scratch onto node-local scratch, not the login filesystem
SCR=/tmp/dmrg_scratch/schur_scaling/${KEY}
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=python3
WORK=benchmarks/response_timing
cd $WORK
echo "scaling $KEY start $(date) host $(hostname)"
$PY run_schur_vs_global_scaling.py --only $KEY --threads 8 \
    --stack-mem-mb 8000 \
    --out $WORK/data/schur_vs_global_${KEY}.json
echo "done $KEY $(date)"
