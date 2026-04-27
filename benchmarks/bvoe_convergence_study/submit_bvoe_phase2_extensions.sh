#!/bin/bash
#SBATCH -J bvoe_p2x
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-12:00:00
#SBATCH -o bvoe_phase2_ext_%j.out
#SBATCH -e bvoe_phase2_ext_%j.err

set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYSCF_PYTHON=${PYSCF_PYTHON:-python}

SYSTEMS=("$@")
if [ "${#SYSTEMS[@]}" -eq 0 ]; then
  SYSTEMS=(c2 lif)
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export BVOE_THREADS=${SLURM_CPUS_PER_TASK}
export BVOE_SWEEPS=${BVOE_SWEEPS:-30}

cd "$ROOT"

echo "Running BVOE phase-2 extension systems: ${SYSTEMS[*]}"
echo "Start: $(date)"
"$PYSCF_PYTHON" run_bvoe_phase2.py "${SYSTEMS[@]}"
"$PYSCF_PYTHON" plot_bvoe_phase2.py
echo "Done: $(date)"
