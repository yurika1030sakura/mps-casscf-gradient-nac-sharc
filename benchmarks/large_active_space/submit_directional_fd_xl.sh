#!/bin/bash
#SBATCH -J dirfd_xl
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=192G
#SBATCH -t 48:00:00
#SBATCH --array=0-1
#SBATCH -o slurm_dirfdxl_%A_%a.out
#SBATCH -e slurm_dirfdxl_%A_%a.err
set -euo pipefail

# Second and third beyond-FCI systems (C20 itself is run by submit_directional_fd.sh).
# Together C20 / C22 / C24 give three beyond-FCI directional-gradient systems.
NC=(22 24)                    # CAS(22,22) det 4.98e11 / CAS(24,24) det 7.31e12
MAXM=(800 800)
N=${NC[$SLURM_ARRAY_TASK_ID]}
M=${MAXM[$SLURM_ARRAY_TASK_ID]}

SCR=/tmp/dmrg_scratch/dirfd/c${N}
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=python3
WORK=benchmarks/large_active_space
cd $WORK
echo "directional FD beyond-FCI C${N} (max M=${M}) start $(date) host $(hostname)"
$PY run_cas_directional_fd.py --ncarbon $N --basis sto-3g \
    --directions bla central_cc single_cc \
    --max-m $M --threads 8 --stack-mem-mb 8000 --h-bohr 1.0e-3 \
    --out $WORK/data/dirfd_c${N}.json
echo "done C${N} $(date)"
