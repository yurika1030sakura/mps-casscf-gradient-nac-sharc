#!/bin/bash
#SBATCH -J anth_pi14_gn
#SBATCH -A woo_lab
#SBATCH -p test
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH -t 0-12:00:00
#SBATCH -o anthracene_pi14_gradnac_%j.out
#SBATCH -e anthracene_pi14_gradnac_%j.err

set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PYSCF_PYTHON=${PYSCF_PYTHON:-python}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd "$ROOT"

echo "Anthracene CAS(14,14) gradient/NAC benchmark start: $(date)"
echo "Python: $PYSCF_PYTHON"

"$PYSCF_PYTHON" -m py_compile run_anthracene_pi14_gradnac.py
"$PYSCF_PYTHON" run_anthracene_pi14_gradnac.py \
  --response-mode "${ANTH_RESPONSE_MODE:-projected-ci}" \
  --basis "${ANTH_BASIS:-sto-3g}" \
  --m-list "${ANTH_GRADNAC_M_LIST:-256,512}" \
  --orbital-m "${ANTH_GRADNAC_ORBITAL_M:-512}" \
  --nroots "${ANTH_NROOTS:-2}" \
  --root-buffer "${ANTH_ROOT_BUFFER:-2}" \
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
  --gradient-states "${ANTH_GRADIENT_STATES:-all}" \
  --nac-pairs "${ANTH_NAC_PAIRS:-0-1}" \
  --response-tol "${ANTH_RESPONSE_TOL:-1e-6}" \
  --response-max-iter "${ANTH_RESPONSE_MAX_ITER:-30}" \
  --response-linear-solver "${ANTH_RESPONSE_LINEAR_SOLVER:-gmres}" \
  --mps-fit-sweeps "${ANTH_MPS_FIT_SWEEPS:-6}" \
  --mps-fit-tol "${ANTH_MPS_FIT_TOL:-1e-7}" \
  ${ANTH_RESPONSE_M_COMPRESS:+--response-m-compress "$ANTH_RESPONSE_M_COMPRESS"} \
  --response-initial-guess "${ANTH_RESPONSE_INITIAL_GUESS:-zero}" \
  --response-initial-guess-sweeps "${ANTH_RESPONSE_INITIAL_GUESS_SWEEPS:-4}" \
  --response-initial-guess-tol "${ANTH_RESPONSE_INITIAL_GUESS_TOL:-1e-6}" \
  --response-initial-guess-proj-weight "${ANTH_RESPONSE_INITIAL_GUESS_PROJ_WEIGHT:-20.0}" \
  --active-localization "${ANTH_ACTIVE_LOCALIZATION:-none}" \
  --active-order "${ANTH_ACTIVE_ORDER:-none}" \
  --mps-coeff-cutoff "${ANTH_MPS_COEFF_CUTOFF:-1e-10}" \
  --lagrange-max-cycle "${ANTH_LAGRANGE_MAX_CYCLE:-500}" \
  --lagrange-conv-atol "${ANTH_LAGRANGE_CONV_ATOL:-1e-10}" \
  --lagrange-conv-rtol "${ANTH_LAGRANGE_CONV_RTOL:-1e-6}" \
  ${ANTH_FCI_REFERENCE:+--fci-reference} \
  ${ANTH_SAVE_REFERENCE_NPZ:+--save-reference-npz} \
  ${ANTH_REUSE_REFERENCE_NPZ:+--reuse-reference-npz} \
  ${ANTH_REFERENCE_NPZ:+--reference-npz "$ANTH_REFERENCE_NPZ"} \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --memory-mb "${ANTH_MEMORY_MB:-180000}" \
  --stack-mem "${ANTH_STACK_MEM:-4000000000}" \
  --scratch-root "${TMPDIR:-/tmp}" \
  --out "${ANTH_GRADNAC_OUT:-data/anthracene_pi14_gradnac.json}"

echo "Anthracene CAS(14,14) gradient/NAC benchmark done: $(date)"
