#!/bin/bash
# CF features -> score with the FROZEN CCinc-neutrino model -> the two MVA decks.
#
# NOTHING IS TRAINED. src/ambe/mc/mva_analysis.py is invoked only with --score-data, against the
# same frozen .pkl and the same 0.423621 threshold already used to select neutrons in
# the AmBe v4 DATA deck. That is what makes the MC and data MVA decks the SAME
# selection rather than two different measurements.
#
# ONE extraction, not two. The prompt veto is a per-event `continue` taken before any
# feature is computed, so the veto-ON cluster table is exactly the veto-OFF table minus
# rows from vetoed events. scripts/workflows/derive_vetoon_scored.py performs that filter using the box
# deck's own per-event is_cosmic verdict, and asserts the result. Extracting twice
# would cost ~11 h of CPU to reproduce a row filter.
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
export PYTHONPATH=src
PY=/exp/annie/app/users/dajana/myboy/bin/python
R=/pnfs/annie/persistent/users/dajana/output/AmBe_MC_toolchain/productionv2/QE_1.50
GEO=/exp/annie/app/users/dajana/EB_BC_TA/configfiles/LoadGeometry
OUT=ambe_output/mc_ambe/mva
MODEL=ambe_output/cc_neutrino_v3_truthtag/parquet/cc_neutrino_v3_truthtag__mva_frozen__keepprompt__merged__cf.pkl

echo "######## CF features, veto OFF (the superset)  $(date) ########"
$PY -u -m scripts.workflows.extract_cf_features_beamcluster_data "$R" \
    --input-kind mc --mc-label-map "$R/label_map.txt" --cosmic-veto off \
    --run-name mcqe150_vetoOFF --output-dir $OUT \
    --geometry $GEO/FullTankPMTGeometry.csv --offsets $GEO/TankPMTTimingOffsets.csv \
    --prompt-window-ns 2000 --jobs 4 || exit 1

echo "######## scoring veto OFF with the FROZEN model  $(date) ########"
$PY -u -m ambe.mc.mva_analysis --score-data $OUT/mcqe150_vetoOFF__data_features_cf.parquet \
    --model $MODEL || exit 1

echo "######## deriving the veto-ON scored table  $(date) ########"
$PY -u -m scripts.workflows.derive_vetoon_scored \
    $OUT/mcqe150_vetoOFF__data_features_cf__scored.parquet \
    $OUT/mcqe150_vetoON__data_features_cf__scored.parquet \
    'TriggerSummary/EventIndex_MCAmBe_QE150_box_vetoON_*.parquet' || exit 1

echo "######## MVA decks  $(date) ########"
$PY -u -m ambe.cli data processmc --config configs/mc_ambe_qe150_mva_vetoon.yaml  --selection mva --cosmic-veto on
$PY -u -m ambe.cli data processmc --config configs/mc_ambe_qe150_mva_vetooff.yaml --selection mva --cosmic-veto off

echo "######## MVA deck plots  $(date) ########"
./run_mc_qe150_plots.sh mc_ambe_qe150_mva_vetoon mc_ambe_qe150_mva_vetooff
echo "ALL DONE $(date)"
