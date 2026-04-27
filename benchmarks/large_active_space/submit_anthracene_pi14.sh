#!/bin/bash
#SBATCH -J anth_pi14
#SBATCH -p test
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 0-12:00:00
#SBATCH -o anthracene_pi14_%j.out
#SBATCH -e anthracene_pi14_%j.err

set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PYSCF_PYTHON=${PYSCF_PYTHON:-python}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$ROOT"

echo "Anthracene CAS(14,14) benchmark start: $(date)"
echo "Python: $PYSCF_PYTHON"

"$PYSCF_PYTHON" -m py_compile run_anthracene_pi14.py
"$PYSCF_PYTHON" run_anthracene_pi14.py \
  --basis "${ANTH_BASIS:-sto-3g}" \
  --m-list "${ANTH_M_LIST:-64,128,256,512,1024}" \
  --nroots "${ANTH_NROOTS:-2}" \
  --sweeps "${ANTH_SWEEPS:-80}" \
  --sweep-tol "${ANTH_SWEEP_TOL:-1e-8}" \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --scratch-root "${TMPDIR:-/tmp}"

echo "Anthracene CAS(14,14) benchmark done: $(date)"
