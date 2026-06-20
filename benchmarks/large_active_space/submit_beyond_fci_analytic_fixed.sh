#!/bin/bash
#SBATCH -J bfanalytF
#SBATCH -A ryl_lab
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=180G
#SBATCH -t 36:00:00
#SBATCH --array=0-1
#SBATCH -o slurm_bfanalytF_%A_%a.out
#SBATCH -e slurm_bfanalytF_%A_%a.err
set -euo pipefail
# Phase-2 FIXED config: cr solver + hcc-inverse preconditioned guess + m_compress=256
# + response tol 1e-4. Ablation partner of the gmres/zero/m=build baseline (23684922).
NC=(18 20)
N=${NC[$SLURM_ARRAY_TASK_ID]}
SCR=/n/netscratch/woo_lab/Lab/yulili/bfanalytF/c${N}
mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11
WORK=/n/home04/yulili/daisuan/dmrg_sacasscf_response_public/benchmarks/large_active_space
cd $WORK
echo "beyond-FCI ANALYTIC response (FIXED cfg) C${N} start $(date) host $(hostname)"
$PY run_beyond_fci_analytic.py --ncarbon $N --bond-dim 800 --threads 8 \
    --stack-mem-mb 16000 --linear-solver cr --initial-guess hcc-inverse \
    --m-compress 256 --response-tol 1e-4 \
    --out $WORK/data/beyond_fci_analytic_fixed_c${N}.json
echo "done C${N} $(date)"
