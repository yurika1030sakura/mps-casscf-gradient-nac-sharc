#!/bin/bash
#SBATCH -J acene
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=240G
#SBATCH -t 24:00:00
#SBATCH -o slurm_acene_%j.out
#SBATCH -e slurm_acene_%j.err
set -euo pipefail
NR=${1:-2}; EXTRA="${2:-}"
SCR=/tmp/dmrg_scratch/acene/n${NR}; mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=python3
cd benchmarks/large_active_space
echo "acene nrings=$NR extra=$EXTRA start $(date) host $(hostname)"
$PY acene_beyond_fci.py --nrings $NR --bond-dim 800 --m-ceiling 800 --threads 16 --stack-mem-mb 24000 $EXTRA
echo "done $(date)"
