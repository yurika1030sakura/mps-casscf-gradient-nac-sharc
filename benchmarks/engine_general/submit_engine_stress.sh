#!/bin/bash
#SBATCH -J engstress
#SBATCH -A ryl_lab
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o slurm_engstress_%j.out
#SBATCH -e slurm_engstress_%j.err
set -euo pipefail

# Generality stress test: run the system-general certified engine, with ZERO
# per-system tuning, across singlet/doublet/triplet sectors and several elements,
# and record each system's PASS/WARN/FAIL verdict + certificate + spin purity.
SCR=/n/netscratch/woo_lab/Lab/yulili/engstress
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=/n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11
WORK=/n/home04/yulili/daisuan/dmrg_sacasscf_response_public/benchmarks/engine_general
cd $WORK
echo "engine stress start $(date) host $(hostname)"
$PY run_engine_stress_test.py --out $WORK/data/engine_stress.json
echo "done $(date)"
