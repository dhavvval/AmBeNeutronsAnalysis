set -u
cd /exp/annie/app/users/dajana/AmBeNeutronsAnalysis
for S in truthtag recotag; do
  echo "################ PILOT $S  $(date) ################"
  RUN=cc_neutrino_v3_${S}_pilot STOP_STAGE=2 METHODS="optics clusterfinder" \
    LOG=logs_pilot_v3_${S}.log bash run_ccinc_xi02_compcut.sh
  echo "---- PILOT $S exit $?  $(date) ----"
done
echo "PILOTS DONE $(date)"
