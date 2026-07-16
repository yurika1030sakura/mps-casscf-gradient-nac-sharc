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
  --m-list "${ANTH_M_LIST:-64,128,256,512,1024,2048}" \
  --nroots "${ANTH_NROOTS:-2}" \
  --sweeps "${ANTH_SWEEPS:-80}" \
  --sweep-tol "${ANTH_SWEEP_TOL:-1e-8}" \
  --fci-solver-roots "${ANTH_FCI_SOLVER_ROOTS:-6}" \
  --fci-conv-tol "${ANTH_FCI_CONV_TOL:-1e-12}" \
  --fci-max-cycle "${ANTH_FCI_MAX_CYCLE:-300}" \
  --fci-max-space "${ANTH_FCI_MAX_SPACE:-80}" \
  --fci-pspace-size "${ANTH_FCI_PSPACE_SIZE:-2000}" \
  --fci-spin-tol "${ANTH_FCI_SPIN_TOL:-1e-6}" \
  --fci-degeneracy-tol "${ANTH_FCI_DEGENERACY_TOL:-1e-4}" \
  --dav-thrd "${ANTH_DAV_THRD:-1e-14}" \
  --dav-max-iter "${ANTH_DAV_MAX_ITER:-8000}" \
  --dav-def-max-size "${ANTH_DAV_DEF_MAX_SIZE:-80}" \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --scratch-root "${TMPDIR:-/tmp}"

echo "Anthracene CAS(14,14) benchmark done: $(date)"
