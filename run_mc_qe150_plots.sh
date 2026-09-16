#!/bin/bash
# All deck plots. Efficiency heatmaps are suppressed via plots.heatmap.maps in the
# configs -- no efficiency is reported from these decks.
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
export PYTHONPATH=src
PY=/exp/annie/app/users/dajana/myboy/bin/python
for CFG in "$@"; do
  TAG=$(grep -m1 '^  tag:' configs/${CFG}.yaml | awk '{print $2}')
  echo "######## $CFG (tag $TAG)  $(date) ########"
  $PY -u -m ambe.cli plots combined --config configs/${CFG}.yaml 2>&1 | grep -v '^ ' | tail -4
  $PY -u -m ambe.cli plots heatmap  --config configs/${CFG}.yaml 2>&1 | grep -v '^ ' | tail -4
  $PY -u -m ambe.cli plots basic    --config configs/${CFG}.yaml 2>&1 | grep -v '^ ' | tail -6
  $PY -u -m ambe.stats.capture_time --mode tagset --tags "$TAG" \
      --out-label "$TAG" --fit-backend lmfit 2>&1 | tail -6
done
echo "PLOTS DONE $(date)"
