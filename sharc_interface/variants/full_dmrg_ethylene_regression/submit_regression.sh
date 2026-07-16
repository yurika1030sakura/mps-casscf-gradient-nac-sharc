#!/bin/bash
#SBATCH -J fdmg_eth
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH -o fdmg_eth_%j.out
#SBATCH -e fdmg_eth_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
COMMON_ROOT=${REPO_ROOT}/sharc_interface
QM_DIR=${SCRIPT_DIR}/QM

export PYSCF_PYTHON=${PYSCF_PYTHON:-python}
export SHARC_PYSCF_SCRIPT=${SHARC_PYSCF_SCRIPT:-${COMMON_ROOT}/SHARC_PYSCF_ext.py}
: "${SLURM_JOB_ID:=local}"
: "${SCRATCH_BASE:=${TMPDIR:-/tmp}/dmrg_sacasscf_sharc}"
export TMPDIR=${TMPDIR:-${SCRATCH_BASE}/full_dmrg_ethylene_${SLURM_JOB_ID}/tmp}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "$TMPDIR" "$QM_DIR/SAVEDIR"
cd "$QM_DIR"
cp -f ../PYSCF.template ../PYSCF.resources .

echo "Full DMRG-CASSCF ethylene SHARC regression starting at $(date)"
"$PYSCF_PYTHON" "$SHARC_PYSCF_SCRIPT" QM.in > QM.log 2> QM.err
echo "Full DMRG-CASSCF ethylene SHARC regression done at $(date)"

grep -E "method dmrg-casscf|Error Codes|Runtime|Traceback|failed|Hamiltonian|Gradient|Nonadiabatic" QM.log QM.out 2>/dev/null | tail -220 || true
