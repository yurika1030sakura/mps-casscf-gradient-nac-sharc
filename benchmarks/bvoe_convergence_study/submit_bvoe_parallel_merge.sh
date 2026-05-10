#!/bin/bash
#SBATCH -J bvoe_merge
#SBATCH -A woo_lab
#SBATCH -p test
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-02:00:00
#SBATCH -o bvoe_parallel_merge_%j.out
#SBATCH -e bvoe_parallel_merge_%j.err

set -euo pipefail

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}
PYSCF_PYTHON=${PYSCF_PYTHON:-python}

cd "$ROOT"

echo "BVOE parallel merge start: $(date)"
"$PYSCF_PYTHON" -m py_compile merge_bvoe_parallel_results.py plot_bvoe_phase2.py audit_bvoe_references.py
"$PYSCF_PYTHON" merge_bvoe_parallel_results.py
"$PYSCF_PYTHON" plot_bvoe_phase2.py
"$PYSCF_PYTHON" audit_bvoe_references.py
echo "BVOE parallel merge done: $(date)"
