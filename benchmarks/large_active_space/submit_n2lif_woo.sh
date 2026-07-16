#!/bin/bash
#SBATCH -J n2lif_woo
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o slurm_n2lif_woo_%j.out
#SBATCH -e slurm_n2lif_woo_%j.err
set -euo pipefail
SCR=/tmp/dmrg_scratch/n2lif_woo2; mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=python3
cd src/dmrg_analytic_dev
echo "N2/LiF FD (gmres) start $(date) host $(hostname)"
$PY test_fd_validation_extended.py
echo "done $(date)"
