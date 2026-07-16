#!/bin/bash
#SBATCH -J ethy_traj
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o slurm_ethy_%j.out
#SBATCH -e slurm_ethy_%j.err
set -euo pipefail

SCR=/tmp/dmrg_scratch/ethy_traj
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=python3
WORK=benchmarks/response_timing
cd $WORK
echo "ethylene torsion trajectory start $(date) host $(hostname)"
$PY run_repeated_call_trajectory.py --n-steps 46 --theta-max 90 \
    --basis 6-31G --bond-dim 200 --threads 8 --stack-mem-mb 6000 \
    --out $WORK/data/repeated_call_trajectory.json
echo "done ethylene trajectory $(date)"
