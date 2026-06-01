#!/bin/bash
#SBATCH -J full_dmrg_h2
#SBATCH -A woo_lab
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o full_dmrg_h2_%j.out
#SBATCH -e full_dmrg_h2_%j.err

set -euo pipefail

COMMON_ROOT=/n/home04/yulili/daisuan/prebiotic_sutherland/sharc_pyscf_casscf
ROOT=${COMMON_ROOT}/variants/full_dmrg_h2_smoke
QM_DIR=${ROOT}/QM

export PYSCF_PYTHON=/n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11
export SHARC_PYSCF_SCRIPT=${COMMON_ROOT}/SHARC_PYSCF_ext.py
: "${SLURM_JOB_ID:=local}"
: "${SCRATCH_BASE:=/n/netscratch/woo_lab/Lab/${USER}/prebiotic_sutherland/sharc_pyscf_casscf}"
export TMPDIR=${TMPDIR:-${SCRATCH_BASE}/full_dmrg_h2_${SLURM_JOB_ID}/tmp}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "$TMPDIR" "$QM_DIR/SAVEDIR"
cd "$QM_DIR"
cp -f ../PYSCF.template ../PYSCF.resources .

echo "Full DMRG-CASSCF SHARC H2 smoke starting at $(date)"
"$PYSCF_PYTHON" "$SHARC_PYSCF_SCRIPT" QM.in > QM.log 2> QM.err
echo "Full DMRG-CASSCF SHARC H2 smoke done at $(date)"

grep -E "method dmrg-casscf|Error Codes|Runtime|Hamiltonian|Gradient|Nonadiabatic|END" QM.log QM.out 2>/dev/null | tail -200 || true
