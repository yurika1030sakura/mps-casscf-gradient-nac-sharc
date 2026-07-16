#!/bin/bash
#SBATCH -J fd_suite
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o slurm_fd_%j.out
#SBATCH -e slurm_fd_%j.err
set -euo pipefail

SCR=/tmp/dmrg_scratch/fd_suite
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

PY=python3
WORK=benchmarks/fd_validation
cd $WORK
echo "FD validation suite start $(date) host $(hostname)"
$PY run_fd_validation_suite.py --system all \
    --h-scan 2e-3 1e-3 5e-4 2e-4 \
    --out $WORK/data/fd_validation_suite.jsonl
echo "done $(date)"
