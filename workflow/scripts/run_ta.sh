#!/bin/bash
# One productionv3fix WCSim file through ToolAnalysis in the anniesoft container.
#   run_ta.sh <tank|world> <part> <out ANNIEEvent.root> <work dir> <TA repo> <image> <wcsim dir> <toolchain> <wcsim lib> <python>
set -eu
s=$1 n=$2 out=$3 W=$4 TA=$5 IMG=$6 WD=$7 TC=$8 WLIB=$9 PY=${10}
d=$W/cfg/${s}_$n in=$W/in/$s/wcsim_0.$n.0.root log=$W/logs/ta_${s}_$n.log
setkey () { $PY - "$1" "$2" "$3" <<'EOF'
import sys
p, k, v = sys.argv[1:]
lines = open(p).read().splitlines(True)
hits = [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#") and l.split()[0] == k]
assert len(hits) == 1, f"{p}: {len(hits)} active '{k}' lines"
lines[hits[0]] = f"{k} {v}\n"
open(p, "w").writelines(lines)
EOF
}
rm -rf $d; cp -r $TA/configfiles/$TC $d
sed -i "s#\./configfiles/$TC/#$d/#g; s#configfiles/$TC/#$d/#g" $d/*
mkdir -p $(dirname $in) $W/logs $(dirname $out)
ln -sf $WD/wcsim_$n.root $in
setkey $d/LoadWCSimConfig InputFile $in
[ $s = tank ] && setkey $d/LoadGenieEventConfig FilePattern gntp.$n.ghep.root
setkey $d/ANNIEEventTreeMakerConfig OutputFile $out
setkey $d/ClusterFinderConfig OutputFile $d/mc_clusters.root
setkey $d/FindMrdTracksConfig OutputDirectory $d
setkey $d/TimeClusteringConfig OutputROOTFile $d/STEC_TimeClusteringOut
echo "$(cd $TA && git log --oneline -1) | $(grep -hE '^(useTimeSmearing|PMTParameterFile)' $d/PMTWaveformSimConfig | tr '\n' ' ')" > $log
LD_LIBRARY_PATH=$WLIB $PY $(dirname $(readlink -f $0))/../../src/ambe/provenance.py snapshot --tag ta_${s}_$n --out-file $W/logs/provenance_ta_${s}_$n.json >> $log 2>&1 || true
set +e
singularity exec -B /exp/annie:/exp/annie -B /pnfs:/pnfs $IMG bash -c \
  "cd $TA; source Setup.sh >/dev/null 2>&1; export LD_LIBRARY_PATH=$WLIB:\$LD_LIBRARY_PATH; ./Analyse $d/ToolChainConfig" >> $log 2>&1
rc=$?
set -e
miss=$(grep -c 'Does Not Exist in factory' $log || true)
ent=$($PY -c "import uproot; print(uproot.open('$out')['Event'].num_entries)" 2>/dev/null || echo 0)
echo "rc=$rc missing_tools=$miss entries=$ent" | tee -a $log
[ $rc = 0 ] && [ $miss = 0 ] && [ $ent -gt 0 ]
