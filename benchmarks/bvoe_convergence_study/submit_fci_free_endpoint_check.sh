#!/bin/bash
#SBATCH -J fci_free_ep
#SBATCH -A woo_lab
#SBATCH -p test
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-04:00:00
#SBATCH -o fci_free_endpoint_%j.out
#SBATCH -e fci_free_endpoint_%j.err

set -euo pipefail

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}
PYSCF_PYTHON=${PYSCF_PYTHON:-python}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$ROOT"

echo "FCI-free endpoint check start: $(date)"
echo "Python: $PYSCF_PYTHON"
echo "Systems: ${FCI_FREE_SYSTEMS:-c2,c2_321g,c2_631g,lif,n2,ethylene}"
echo "Default M: ${FCI_FREE_DEFAULT_M:-200}; C2 M: ${FCI_FREE_C2_M:-400}"
echo "Root buffer: ${FCI_FREE_ROOT_BUFFER:-4}"

"$PYSCF_PYTHON" -m py_compile run_fci_free_endpoint_check.py run_bvoe_phase2.py
"$PYSCF_PYTHON" run_fci_free_endpoint_check.py \
  --systems "${FCI_FREE_SYSTEMS:-c2,c2_321g,c2_631g,lif,n2,ethylene}" \
  --default-m "${FCI_FREE_DEFAULT_M:-200}" \
  --c2-m "${FCI_FREE_C2_M:-400}" \
  --root-buffer "${FCI_FREE_ROOT_BUFFER:-4}" \
  --sweeps "${FCI_FREE_SWEEPS:-100}" \
  --sweep-tol "${FCI_FREE_SWEEP_TOL:-1e-14}" \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --out "${FCI_FREE_OUT:-data_fci_free_endpoint/endpoint_check.json}"

echo "FCI-free endpoint check done: $(date)"
