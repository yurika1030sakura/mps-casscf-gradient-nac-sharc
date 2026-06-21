#!/bin/bash
#SBATCH -J n2lif_woo
#SBATCH -A woo_lab
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /n/home04/yulili/daisuan/dmrg_sacasscf_response_public/slurm_n2lif_woo_%j.out
#SBATCH -e /n/home04/yulili/daisuan/dmrg_sacasscf_response_public/slurm_n2lif_woo_%j.err
set -euo pipefail
SCR=/n/netscratch/woo_lab/Lab/yulili/n2lif_woo; mkdir -p $SCR
export TMPDIR=$SCR TMP=$SCR TEMP=$SCR PYSCF_TMPDIR=$SCR OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11
cd /n/home04/yulili/daisuan/dmrg_sacasscf_response_public/src/dmrg_analytic_dev
echo "N2/LiF FD (cr,8thr) start $(date) host $(hostname)"
$PY test_fd_validation_extended.py
echo "done $(date)"
