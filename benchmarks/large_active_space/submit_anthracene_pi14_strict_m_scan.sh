#!/bin/bash
# Submit a strict-response M scan for the anthracene CAS(14,14) MPS-Krylov
# benchmark.  This is the production-style companion to the loose/default M
# scan: DMRG roots and response vectors remain MPS-native, while FCI data are
# used only for post hoc validation.

set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PUBLIC_ROOT=${PUBLIC_ROOT:-$(cd -- "$ROOT/../.." && pwd)}
BENCH_ROOT=${BENCH_ROOT:-$ROOT}
PYSCF_PYTHON=${PYSCF_PYTHON:-python}

PARTITION=${ANTH_STRICT_SCAN_PARTITION:-test}
TIME_LIMIT=${ANTH_STRICT_SCAN_TIME:-0-12:00:00}
CPUS=${ANTH_STRICT_SCAN_CPUS:-8}
MEM=${ANTH_STRICT_SCAN_MEM:-180G}
M_LIST_TEXT=${ANTH_STRICT_SCAN_M_LIST:-"64 128 256"}
ORBITAL_M=${ANTH_STRICT_SCAN_ORBITAL_M:-512}

RESPONSE_TOL=${ANTH_STRICT_RESPONSE_TOL:-1e-8}
RESPONSE_MAX_ITER=${ANTH_STRICT_RESPONSE_MAX_ITER:-120}
MPS_FIT_SWEEPS=${ANTH_STRICT_MPS_FIT_SWEEPS:-8}
MPS_FIT_TOL=${ANTH_STRICT_MPS_FIT_TOL:-1e-8}
RESPONSE_M_COMPRESS=${ANTH_STRICT_RESPONSE_M_COMPRESS:-}

SUBMIT_SCRIPT="$ROOT/submit_anthracene_pi14_gradnac.sh"
COMBINE_SCRIPT="$ROOT/combine_partial_derivatives.py"
REFERENCE_NPZ=${ANTH_REFERENCE_NPZ:-$BENCH_ROOT/data/anthracene_pi14_reference_cache.npz}

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "missing submit script: $SUBMIT_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$COMBINE_SCRIPT" ]]; then
  echo "missing combine script: $COMBINE_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$REFERENCE_NPZ" ]]; then
  echo "missing cached FCI/CASSCF reference: $REFERENCE_NPZ" >&2
  echo "Run run_anthracene_pi14_gradnac.py once with --save-reference-npz first." >&2
  exit 1
fi

mkdir -p "$BENCH_ROOT/data"

submit_partial() {
  local m=$1
  local task=$2
  local grad_states
  local nac_pairs
  local suffix

  case "$task" in
    g0) grad_states=0; nac_pairs=none; suffix=grad0 ;;
    g1) grad_states=1; nac_pairs=none; suffix=grad1 ;;
    nac) grad_states=none; nac_pairs=0-1; suffix=nac01 ;;
    *) echo "unknown task: $task" >&2; exit 1 ;;
  esac

  local out="$BENCH_ROOT/data/anthracene_pi14_mps_native_M${m}_strict_scan_${suffix}.json"
  local export_vars
  export_vars="ALL"
  export_vars+=",PYSCF_PYTHON=$PYSCF_PYTHON"
  export_vars+=",ANTH_GRADNAC_M_LIST=$m"
  export_vars+=",ANTH_GRADNAC_ORBITAL_M=$ORBITAL_M"
  export_vars+=",ANTH_REFERENCE_NPZ=$REFERENCE_NPZ"
  export_vars+=",ANTH_REUSE_REFERENCE_NPZ=1"
  export_vars+=",ANTH_ACTIVE_LOCALIZATION=boys"
  export_vars+=",ANTH_ACTIVE_ORDER=principal-axis"
  export_vars+=",ANTH_GRADIENT_STATES=$grad_states"
  export_vars+=",ANTH_NAC_PAIRS=$nac_pairs"
  export_vars+=",ANTH_RESPONSE_MODE=mps-krylov"
  export_vars+=",ANTH_RESPONSE_TOL=$RESPONSE_TOL"
  export_vars+=",ANTH_RESPONSE_MAX_ITER=$RESPONSE_MAX_ITER"
  export_vars+=",ANTH_MPS_FIT_SWEEPS=$MPS_FIT_SWEEPS"
  export_vars+=",ANTH_MPS_FIT_TOL=$MPS_FIT_TOL"
  if [[ -n "$RESPONSE_M_COMPRESS" ]]; then
    export_vars+=",ANTH_RESPONSE_M_COMPRESS=$RESPONSE_M_COMPRESS"
  fi
  export_vars+=",ANTH_RESPONSE_INITIAL_GUESS=zero"
  export_vars+=",ANTH_GRADNAC_OUT=$out"

  sbatch --parsable \
    --job-name="anth14_s${m}_${task}" \
    --partition="$PARTITION" \
    --time="$TIME_LIMIT" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --export="$export_vars" \
    "$SUBMIT_SCRIPT"
}

submit_combine() {
  local m=$1
  shift
  local deps=$*
  local out="$BENCH_ROOT/data/anthracene_pi14_mps_native_M${m}_strict_scan_combined.json"
  local grad0="$BENCH_ROOT/data/anthracene_pi14_mps_native_M${m}_strict_scan_grad0.json"
  local grad1="$BENCH_ROOT/data/anthracene_pi14_mps_native_M${m}_strict_scan_grad1.json"
  local nac="$BENCH_ROOT/data/anthracene_pi14_mps_native_M${m}_strict_scan_nac01.json"

  sbatch --parsable \
    --job-name="anth14_s${m}_combine" \
    --partition="$PARTITION" \
    --time="0-00:20:00" \
    --cpus-per-task=1 \
    --mem=4G \
    --dependency="afterok:${deps// /:}" \
    --wrap="cd '$ROOT' && '$PYSCF_PYTHON' '$COMBINE_SCRIPT' --out '$out' '$grad0' '$grad1' '$nac'"
}

echo "Submitting anthracene strict-response M scan"
echo "  M list: $M_LIST_TEXT"
echo "  partition: $PARTITION"
echo "  response: tol=$RESPONSE_TOL max_iter=$RESPONSE_MAX_ITER fit_sweeps=$MPS_FIT_SWEEPS fit_tol=$MPS_FIT_TOL"
if [[ -n "$RESPONSE_M_COMPRESS" ]]; then
  echo "  response_m_compress: $RESPONSE_M_COMPRESS"
fi

for m in $M_LIST_TEXT; do
  ids=()
  for task in g0 g1 nac; do
    jid=$(submit_partial "$m" "$task")
    ids+=("$jid")
    echo "  submitted M=$m $task: $jid"
  done
  combine_jid=$(submit_combine "$m" "${ids[@]}")
  echo "  submitted M=$m combine: $combine_jid after ${ids[*]}"
done
