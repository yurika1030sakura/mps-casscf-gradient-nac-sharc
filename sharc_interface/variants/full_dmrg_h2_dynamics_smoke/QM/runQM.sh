#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

: "${PYSCF_PYTHON:=/n/holylabs/woo_lab/Lab/yulili_pyscf/env/bin/python3.11}"

if [ -z "${SHARC_PYSCF_SCRIPT:-}" ]; then
  if [ -f ../SHARC_PYSCF_ext.py ]; then
    SHARC_PYSCF_SCRIPT=../SHARC_PYSCF_ext.py
  elif [ -f ./SHARC_PYSCF_ext.py ]; then
    SHARC_PYSCF_SCRIPT=./SHARC_PYSCF_ext.py
  else
    echo "Could not locate SHARC_PYSCF_ext.py from $SCRIPT_DIR" >&2
    exit 1
  fi
fi

"$PYSCF_PYTHON" "$SHARC_PYSCF_SCRIPT" QM.in >> QM.log 2>> QM.err
