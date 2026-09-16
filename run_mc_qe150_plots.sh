#!/bin/bash
# Write new products in this checkout; archived inputs remain configurable.
AMBE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export AMBE_OUT="${AMBE_OUT:-$AMBE_REPO/ambe_output}"
# All deck plots. Efficiency heatmaps are suppressed via plots.heatmap.maps in the
# configs -- no efficiency is reported from these decks.
cd "$AMBE_REPO" || exit 1
export PYTHONPATH=src
PY=/exp/annie/app/users/dajana/myboy/bin/python
for CFG in "$@"; do
  echo "######## $CFG (tag $TAG)  $(date) ########"
  $PY -u -m ambe.cli plots combined --config configs/${CFG}.yaml 2>&1 | grep -v '^ ' | tail -4
  $PY -u -m ambe.cli plots heatmap  --config configs/${CFG}.yaml 2>&1 | grep -v '^ ' | tail -4
  $PY -u -m ambe.cli plots basic    --config configs/${CFG}.yaml 2>&1 | grep -v '^ ' | tail -6
done
echo "PLOTS DONE $(date)"
