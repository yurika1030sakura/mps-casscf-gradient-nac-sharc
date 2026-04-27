#!/bin/bash
#SBATCH -J bvoe_highm
#SBATCH -A woo_lab
#SBATCH -p test
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-12:00:00
#SBATCH -o highm_public_scan_%j.out
#SBATCH -e highm_public_scan_%j.err

set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
PYSCF_PYTHON=${PYSCF_PYTHON:-python}

# The benchmark code caches the FCI reference orbitals/CI vectors and uses
# that fixed gauge for every M point, so threaded DMRG runs remain comparable
# against the same reference.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export BVOE_THREADS=${SLURM_CPUS_PER_TASK:-8}
export BVOE_SWEEPS=${BVOE_SWEEPS:-100}
export BVOE_ROOT_BUFFER=${BVOE_ROOT_BUFFER:-4}

cd "$ROOT"

echo "High-M public BVOE scan start: $(date)"
echo "Python: $PYSCF_PYTHON"
echo "BVOE_SWEEPS=$BVOE_SWEEPS BVOE_THREADS=$BVOE_THREADS BVOE_ROOT_BUFFER=$BVOE_ROOT_BUFFER"

"$PYSCF_PYTHON" -m py_compile run_bvoe_phase2.py plot_bvoe_phase2.py
"$PYSCF_PYTHON" run_bvoe_phase2.py
"$PYSCF_PYTHON" plot_bvoe_phase2.py

echo "High-M public BVOE scan done: $(date)"
