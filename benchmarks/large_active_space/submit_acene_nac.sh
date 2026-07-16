#!/bin/bash
#SBATCH -J acenenac
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=360G
#SBATCH -t 36:00:00
#SBATCH -o slurm_acenenac_%j.out
#SBATCH -e slurm_acenenac_%j.err
set -euo pipefail
NR=${1:-2}; EXTRA="${2:-}"
SCR=/tmp/dmrg_scratch/acenenac/n${NR}; mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=python3
cd benchmarks/large_active_space
echo "acene NAC nrings=$NR extra=$EXTRA start $(date) host $(hostname)"
$PY acene_nac.py --nrings $NR --bond-dim 800 --m-ceiling 800 --threads 16 --stack-mem-mb 24000 $EXTRA
echo "done $(date)"
