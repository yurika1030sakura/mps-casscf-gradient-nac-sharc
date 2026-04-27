#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 PATH_TO_PRIVATE_TERM_LIST" >&2
  exit 2
fi

term_file=$1
if [ ! -f "$term_file" ]; then
  echo "Term list not found: $term_file" >&2
  exit 2
fi

pattern=$(grep -v '^[[:space:]]*$' "$term_file" | sed 's/[.[\*^$()+?{}|\\]/\\&/g' | paste -sd'|' -)
if [ -z "$pattern" ]; then
  echo "No terms in $term_file"
  exit 0
fi

set +e
rg -n --hidden --glob '!/.git/**' --glob '!/.private_terms' "$pattern" .
status=$?
set -e

if [ "$status" -eq 0 ]; then
  echo "Sensitive-term scan found matches. Fix them before upload." >&2
  exit 1
fi

echo "Sensitive-term scan passed."
