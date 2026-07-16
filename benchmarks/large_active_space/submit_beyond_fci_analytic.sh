#!/bin/bash
#SBATCH -J bfanalytic
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=180G
#SBATCH -t 36:00:00
#SBATCH --array=0-1
#SBATCH -o slurm_bfanalytic_%A_%a.out
#SBATCH -e slurm_bfanalytic_%A_%a.err
set -euo pipefail

# Critical experiment: does the certified ANALYTIC MPS response z-solve complete
# past the FCI wall?  C18 (det 2.36e9, FCI vector ~19 GB) and C20 (det 3.41e10).
NC=(18 20)
N=${NC[$SLURM_ARRAY_TASK_ID]}
SCR=/tmp/dmrg_scratch/bfanalytic/c${N}
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=python3
WORK=benchmarks/large_active_space
cd $WORK
echo "beyond-FCI ANALYTIC response C${N} start $(date) host $(hostname)"
$PY run_beyond_fci_analytic.py --ncarbon $N --bond-dim 800 --threads 8 \
    --stack-mem-mb 16000 --out $WORK/data/beyond_fci_analytic_c${N}.json
echo "done C${N} $(date)"
