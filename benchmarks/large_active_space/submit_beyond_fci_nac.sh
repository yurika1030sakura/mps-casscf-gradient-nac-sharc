#!/bin/bash
#SBATCH -J polyene_bfnac
#SBATCH -A ryl_lab
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=128G
#SBATCH -t 36:00:00
#SBATCH --array=0-2
#SBATCH -o slurm_bfnac_%A_%a.out
#SBATCH -e slurm_bfnac_%A_%a.err
set -euo pipefail

NC=(10 14 20)               # CAS(10,10) FCI-checked -> CAS(14,14) bridge -> CAS(20,20) beyond FCI
N=${NC[$SLURM_ARRAY_TASK_ID]}

SCR=/n/netscratch/woo_lab/Lab/yulili/polyene_bfnac/c${N}
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=/n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11
WORK=/n/home04/yulili/daisuan/dmrg_sacasscf_response_public/benchmarks/large_active_space
cd $WORK
echo "polyene beyond-FCI NAC C${N} start $(date) host $(hostname)"
$PY run_beyond_fci_nac.py --ncarbon $N --basis sto-3g \
    --bond-dim 800 --threads 8 --stack-mem-mb 8000 --h-bohr 1.0e-3 \
    --out $WORK/data/beyond_fci_nac_c${N}.json
echo "done C${N} $(date)"
