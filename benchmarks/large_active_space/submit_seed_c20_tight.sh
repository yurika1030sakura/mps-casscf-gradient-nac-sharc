#!/bin/bash
#SBATCH -J seedc20t
#SBATCH -A ryl_lab
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=160G
#SBATCH -t 72:00:00
#SBATCH -o slurm_seedc20tight_%j.out
#SBATCH -e slurm_seedc20tight_%j.err
set -euo pipefail

# Does tightening the schedule collapse the C20 seed-3 anomaly?
# Current settings: 3 seeds give e0 = -755.281495 (x2, agree to 1e-8) and
# -755.278629 (seed 3, 2.9 mEh higher) -- a higher local SA-CASSCF stationary
# point.  If the TIGHT schedule (M up to 800, 150 macros, conv 1e-6) drives all
# seeds to the lowest, the spread was under-convergence; if not, it is a genuine
# near-degenerate orbital landscape -> production protocol = multi-seed + lowest.
SCR=/n/netscratch/woo_lab/Lab/yulili/seedindep/c20_tight
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

PY=/n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11
WORK=/n/home04/yulili/daisuan/dmrg_sacasscf_response_public/benchmarks/large_active_space
cd $WORK
echo "seed-independence C20 tight start $(date) host $(hostname)"
$PY run_seed_independence.py --ncarbon 20 --schedule tight --seeds 1 2 3 \
    --threads 8 --stack-mem-mb 8000 \
    --out $WORK/data/seedindep_c20_tight.json
echo "done C20 tight $(date)"
