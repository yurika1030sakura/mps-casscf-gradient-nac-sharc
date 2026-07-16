#!/bin/bash
#SBATCH -J polycont
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=480G
#SBATCH -t 48:00:00
#SBATCH -o slurm_polycont_%j.out
#SBATCH -e slurm_polycont_%j.err
set -euo pipefail
NC=${1:-22}
SCR=/tmp/dmrg_scratch/polycont/c${NC}; mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=python3
cd benchmarks/large_active_space
echo "polyene continuation 承重 C${NC} start $(date) host $(hostname)"
$PY polyene_continuation_fd.py --ncarbon $NC --directions central_cc --seeds 1 2 --K 3 \
    --bond-dim 800 --m-ceiling 800 --threads 16 --stack-mem-mb 24000
echo "done C${NC} $(date)"
