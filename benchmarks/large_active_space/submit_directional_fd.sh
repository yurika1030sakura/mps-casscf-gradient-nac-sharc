#!/bin/bash
#SBATCH -J dirfd
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH --array=0-2
#SBATCH -o slurm_dirfd_%A_%a.out
#SBATCH -e slurm_dirfd_%A_%a.err
set -euo pipefail

NC=(10 14 20)                 # CAS(10,10) FCI-checkable / (14,14) bridge / (20,20) beyond FCI
MAXM=(256 512 800)            # progressive-M ceiling per task
N=${NC[$SLURM_ARRAY_TASK_ID]}
M=${MAXM[$SLURM_ARRAY_TASK_ID]}

SCR=/tmp/dmrg_scratch/dirfd/c${N}
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=python3
WORK=benchmarks/large_active_space
cd $WORK
echo "directional FD C${N} (max M=${M}) start $(date) host $(hostname)"
$PY run_cas_directional_fd.py --ncarbon $N --basis sto-3g \
    --directions bla central_cc single_cc \
    --max-m $M --threads 8 --stack-mem-mb 8000 --h-bohr 1.0e-3 \
    --out $WORK/data/dirfd_c${N}.json
echo "done C${N} $(date)"
