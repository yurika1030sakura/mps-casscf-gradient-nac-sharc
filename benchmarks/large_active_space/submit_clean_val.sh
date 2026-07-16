#!/bin/bash
#SBATCH -J cleanval
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=480G
#SBATCH -t 72:00:00
#SBATCH -o slurm_cleanval_%j.out
#SBATCH -e slurm_cleanval_%j.err
set -euo pipefail
# usage: sbatch --export=ALL,NC=20,SEEDS="1 2 3 4 5" submit_clean_val.sh
NC=${NC:-20}; SEEDS=${SEEDS:-"1 2 3 4 5"}
SCR=/tmp/dmrg_scratch/cleanval/c${NC}; mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=python3
cd benchmarks/large_active_space
echo "clean validation C${NC} seeds=[$SEEDS] start $(date) host $(hostname)"
$PY beyond_fci_validation_clean.py --ncarbon $NC --bond-dim 800 --m-ceiling 800 \
    --ci-m-loop 256 --orb-tol 1e-4 --orb-max-iter 20 --ci-sweeps 16 --ci-tol 1e-6 \
    --h-bohr 1e-3 --ladder-K 4 --seeds $SEEDS --threads 16 --stack-mem-mb 24000
echo "done C${NC} $(date)"
