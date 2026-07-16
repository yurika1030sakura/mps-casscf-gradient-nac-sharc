#!/bin/bash
#SBATCH -J seedindep
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH --array=0-3
#SBATCH -o slurm_seedindep_%A_%a.out
#SBATCH -e slurm_seedindep_%A_%a.err
set -euo pipefail

# Diagnose whether the SA-DMRG-CASSCF reference is seed-independent, and whether
# tighter convergence makes it so.  C14 = near-exact control; C16 = beyond-FCI,
# current vs tight; C20 = the centerpiece, current settings.
NC=(14 16 16 20)
SCHED=(current current tight current)
N=${NC[$SLURM_ARRAY_TASK_ID]}
S=${SCHED[$SLURM_ARRAY_TASK_ID]}

SCR=/tmp/dmrg_scratch/seedindep/c${N}_${S}
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=python3
WORK=benchmarks/large_active_space
cd $WORK
echo "seed-independence C${N} ${S} start $(date) host $(hostname)"
$PY run_seed_independence.py --ncarbon $N --schedule $S --seeds 1 2 3 \
    --threads 8 --stack-mem-mb 8000 \
    --out $WORK/data/seedindep_c${N}_${S}.json
echo "done C${N} ${S} $(date)"
