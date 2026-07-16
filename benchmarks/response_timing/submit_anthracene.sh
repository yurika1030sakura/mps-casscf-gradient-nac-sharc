#!/bin/bash
#SBATCH -J anthracene_h2h
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o slurm_anthracene_%j.out
#SBATCH -e slurm_anthracene_%j.err
set -euo pipefail

SCR=/tmp/dmrg_scratch/anthracene_h2h
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=python3
WORK=benchmarks/response_timing
cd $WORK
echo "anthracene head-to-head start $(date) host $(hostname)"
# M=256: tractable regime where BOTH the global MPS-Krylov solver and the
# sweep-localized Schur solver finish, exposing the cost ratio and the Schur
# orbital-GMRES iteration count.  The M=512 strict global reference (3.67e4 s)
# is already reported in the manuscript table.
$PY run_anthracene_headtohead.py --bond-dim 256 --threads 8 \
    --stack-mem-mb 8000 --max-iter 60 \
    --out $WORK/data/anthracene_headtohead.json
echo "done anthracene $(date)"
