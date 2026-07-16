#!/bin/bash
#SBATCH -J c20scbasin
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=360G
#SBATCH -t 48:00:00
#SBATCH -o slurm_c20scbasin_%j.out
#SBATCH -e slurm_c20scbasin_%j.err
set -euo pipefail
SCR=/tmp/dmrg_scratch/c20scbasin; mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=python3
cd benchmarks/large_active_space
echo "C20 single_cc basin-tracked FD start $(date) host $(hostname)"
$PY c20_single_cc_basin_tracked_fd.py
echo "done $(date)"
