#!/bin/bash
#SBATCH -J anth_pi14_fci
#SBATCH -A woo_lab
#SBATCH -p test
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH -t 0-12:00:00
#SBATCH -o anthracene_pi14_fci_validation_%j.out
#SBATCH -e anthracene_pi14_fci_validation_%j.err

set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PYSCF_PYTHON=${PYSCF_PYTHON:-python}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$ROOT"

echo "Anthracene CAS(14,14) FCI-validation gradient/NAC benchmark start: $(date)"
echo "Python: $PYSCF_PYTHON"

"$PYSCF_PYTHON" -m py_compile run_anthracene_pi14_gradnac.py
"$PYSCF_PYTHON" run_anthracene_pi14_gradnac.py \
  --fci-reference \
  --basis "${ANTH_BASIS:-sto-3g}" \
  --m-list "${ANTH_FCI_VALIDATION_M_LIST:-256,512}" \
  --orbital-m "${ANTH_FCI_VALIDATION_ORBITAL_M:-512}" \
  --nroots "${ANTH_NROOTS:-2}" \
  --root-buffer "${ANTH_ROOT_BUFFER:-4}" \
  --casscf-max-cycle "${ANTH_CASSCF_MAX_CYCLE:-20}" \
  --casscf-conv-tol "${ANTH_CASSCF_CONV_TOL:-1e-8}" \
  --casscf-conv-tol-grad "${ANTH_CASSCF_CONV_TOL_GRAD:-3e-5}" \
  --casscf-nsteps "${ANTH_CASSCF_NSTEPS:-24}" \
  --casscf-sweep-tol "${ANTH_CASSCF_SWEEP_TOL:-1e-7}" \
  --eval-sweeps "${ANTH_EVAL_SWEEPS:-80}" \
  --eval-sweep-tol "${ANTH_EVAL_SWEEP_TOL:-1e-8}" \
  --dav-thrd "${ANTH_DAV_THRD:-1e-12}" \
  --dav-max-iter "${ANTH_DAV_MAX_ITER:-4000}" \
  --dav-def-max-size "${ANTH_DAV_DEF_MAX_SIZE:-80}" \
  --mps-coeff-cutoff "${ANTH_MPS_COEFF_CUTOFF:-1e-10}" \
  --lagrange-max-cycle "${ANTH_LAGRANGE_MAX_CYCLE:-500}" \
  --lagrange-conv-atol "${ANTH_LAGRANGE_CONV_ATOL:-1e-10}" \
  --lagrange-conv-rtol "${ANTH_LAGRANGE_CONV_RTOL:-1e-6}" \
  --fci-solver-roots "${ANTH_FCI_SOLVER_ROOTS:-6}" \
  --fci-conv-tol "${ANTH_FCI_CONV_TOL:-1e-12}" \
  --fci-max-cycle "${ANTH_FCI_MAX_CYCLE:-300}" \
  --fci-max-space "${ANTH_FCI_MAX_SPACE:-80}" \
  --fci-pspace-size "${ANTH_FCI_PSPACE_SIZE:-2000}" \
  --fci-spin-tol "${ANTH_FCI_SPIN_TOL:-1e-6}" \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --memory-mb "${ANTH_MEMORY_MB:-180000}" \
  --stack-mem "${ANTH_STACK_MEM:-4000000000}" \
  --scratch-root "${TMPDIR:-/tmp}" \
  --out "${ANTH_FCI_VALIDATION_OUT:-data/anthracene_pi14_fci_validation_gradnac.json}"

echo "Anthracene CAS(14,14) FCI-validation gradient/NAC benchmark done: $(date)"
