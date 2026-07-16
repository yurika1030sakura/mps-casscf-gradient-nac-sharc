#!/bin/bash
#SBATCH -J lif_cas66
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o slurm_lif66_%j.out
#SBATCH -e slurm_lif66_%j.err
set -euo pipefail

SCR=/tmp/dmrg_scratch/lif_cas66
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=python3
WORK=benchmarks/root_tracking
cd $WORK
echo "LiF CAS(6,6) root tracking start $(date) host $(hostname)"

# Dense scan (loose-to-medium convergence), walltime-safe + resumable.
$PY run_lif_cas66_root_tracking.py \
    --basis 6-31G --ncas 6 --nelecas 6 --nroots 2 \
    --conv-tol-grad 1e-6 --max-cycle-macro 120 \
    --out $WORK/data/lif_cas66_root_tracking.jsonl \
    --resume --walltime-min 720 --walltime-buffer-min 20

# Tight spot checks at the gap-minimum region.
$PY run_lif_cas66_root_tracking.py \
    --basis 6-31G --ncas 6 --nelecas 6 --nroots 2 \
    --R-list 3.35 3.40 3.45 --conv-tol-grad 1e-7 --max-cycle-macro 200 \
    --out $WORK/data/lif_cas66_root_tracking_tight.jsonl \
    --resume --walltime-min 720 --walltime-buffer-min 20

echo "done LiF CAS(6,6) $(date)"
