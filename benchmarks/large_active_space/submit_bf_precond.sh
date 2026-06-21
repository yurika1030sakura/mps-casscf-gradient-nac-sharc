#!/bin/bash
#SBATCH -J bfprec
#SBATCH -A woo_lab
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=180G
#SBATCH -t 48:00:00
#SBATCH --array=0-1
#SBATCH -o /n/home04/yulili/daisuan/dmrg_sacasscf_response_public/slurm_bfprec_%A_%a.out
#SBATCH -e /n/home04/yulili/daisuan/dmrg_sacasscf_response_public/slurm_bfprec_%A_%a.err
set -euo pipefail
NC=(18 20)
N=${NC[$SLURM_ARRAY_TASK_ID]}
SCR=/n/netscratch/woo_lab/Lab/yulili/bfprec/c${N}; mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11
cd /n/home04/yulili/daisuan/dmrg_sacasscf_response_public/benchmarks/large_active_space
echo "bf precond C${N} start $(date) host $(hostname)"
$PY run_beyond_fci_analytic.py --ncarbon $N --bond-dim 512 --threads 8 \
    --stack-mem-mb 16000 --linear-solver cr --initial-guess hcc-inverse \
    --m-compress 128 --response-tol 1e-4 --faulthandler-s 300 --max-iter 120 \
    --out /n/home04/yulili/daisuan/dmrg_sacasscf_response_public/benchmarks/large_active_space/data/beyond_fci_analytic_prec_c${N}.json
echo "done C${N} $(date)"
