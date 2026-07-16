#!/bin/bash
#SBATCH -J fdmg_h2_dyn
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o fdmg_h2_dyn_%j.out
#SBATCH -e fdmg_h2_dyn_%j.err

set -euo pipefail

COMMON_ROOT=sharc_interface
ROOT=${COMMON_ROOT}/variants/full_dmrg_h2_dynamics_smoke

export SHARC_ROOT=/path/to/sharc
export SHARC=/path/to/sharc/source
export SHARC_BIN=/path/to/sharc/bin
export PYSCF_PYTHON=python3
export SHARC_PYSCF_SCRIPT=${COMMON_ROOT}/SHARC_PYSCF_ext.py
export PATH=$SHARC:$SHARC_BIN:$PATH

: "${SLURM_JOB_ID:=local}"
: "${SCRATCH_BASE:=/tmp/dmrg_scratch${USER}/prebiotic_sutherland/sharc_pyscf_casscf}"
export TMPDIR=${SCRATCH_BASE}/full_dmrg_h2_dynamics_${SLURM_JOB_ID}/tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "$TMPDIR" "$ROOT/QM"
cd "$ROOT"
rm -rf QM/SAVEDIR SAVEDIR restart restart.* output.dat output.lis output.log output.xyz sharc.log
mkdir -p QM/SAVEDIR
rm -f QM/QM.in QM/QM.out QM/QM.log QM/QM.err QM/PySCF_*.log
cp -f PYSCF.template PYSCF.resources "${COMMON_ROOT}/runQM.sh" QM/
chmod +x QM/runQM.sh

echo "Full DMRG-CASSCF H2 SHARC dynamics smoke starting at $(date)"
echo "sharc.x: $(which sharc.x)"
echo "PySCF: $PYSCF_PYTHON"
sharc.x input > sharc.log 2>&1
echo "Full DMRG-CASSCF H2 SHARC dynamics smoke done at $(date)"

python3 "${COMMON_ROOT}/summarize_sharc_smoke.py" "$ROOT" \
  --label full_dmrg_h2_dynamics \
  --out "$ROOT/sharc_dynamics_summary.json"
