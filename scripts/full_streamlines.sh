# Write new products in this checkout; archived inputs remain configurable.
AMBE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export AMBE_OUT="${AMBE_OUT:-$AMBE_REPO/ambe_output}"
set -u
cd "$AMBE_REPO" || exit 1
for S in truthtag recotag; do
  echo "################ FULL $S  START $(date) ################"
  RUN=cc_neutrino_v3_${S} STOP_STAGE=2 METHODS="optics clusterfinder" \
    LOG=logs_full_v3_${S}.log bash run_ccinc_xi02_compcut.sh
  rc=$?
  echo "################ FULL $S  EXIT $rc  $(date) ################"
  [ $rc -ne 0 ] && echo "ABORTING CHAIN: $S failed" && exit $rc
done
echo "FULL STREAMLINES DONE $(date)"
