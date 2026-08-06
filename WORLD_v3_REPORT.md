# World-volume productionv3 — first look (2026-07-30)

Sample: /pnfs/annie/persistent/users/dajana/output/genie_wcsim_world/productionv3/world/wcsim_0.root
Ntuple: EB_BC_TA/ANNIEEvent_bt_v3world.root  (config CC_MC_RECO_ntuple_bt_world, staged as wcsim_0.93.0.root)
Analysis config: AmBeNeutronsAnalysis/configs/cc_neutrino_v3world.yaml
Scripts: /exp/annie/data/users/dajana/analyse_v3world_truth.py , analyse_v3world_inout.py

## BLOCKER FOUND AND FIXED: libWCSimRoot.so at runtime
Setup.sh alone makes ./Analyse resolve the container stock ToolDAQ/WCSimLib/libWCSimRoot.so,
which has ZERO DirectParentID symbols -> all per-hit ancestry garbage (92.6% untraced).
A control re-run of the KNOWN-GOOD tank pilot wcsim file failed identically (92.29%), proving
it was the environment, not the world sample. Fix: prepend /exp/annie/app/users/dajana/WCSim/WCSim
to LD_LIBRARY_PATH. After that the world ntuple PASSES validate_lineage_ntuple.py:
100.0000% complete chains, 0 truncated, 0% untraced, photon class 20.35%, depth mean 1.96 max 10.

## STILL BROKEN: the ntuple GENIE truth branches on this sample
corr(flux nuvtx, trueNuIntxVtx) ~ 0 in all three coordinates -> LoadGenieEvent is matched to the
wrong events. Tank pilot for comparison: corr(trueMuonEnergy,trueFSLEnergy)=+1.0000.
So trueCC / trueNeutrons / trueNuIntxVtx / trueFSL* must NOT be used from ANNIEEvent_bt_v3world.root.
Inside/outside truth was taken instead from the g4dirt tank-flux file (authoritative);
flux entry i <-> ntuple row i verified: corr(KE into tank, nhits)=+0.60 aligned, ~0.00 for any shift.

> ### SCOPE CORRECTION (2026-08-04) — this applies ONLY to the 7/30 hand-built ntuple
>
> The "GENIE truth is broken" finding above is about **`EB_BC_TA/ANNIEEvent_bt_v3world.root`**
> (built 2026-07-30 from `world/wcsim_0.root`). It does **NOT** apply to the later grid
> production at
> `/pnfs/annie/persistent/users/dajana/output/genie_wcsim_world/productionv3/world/fmvmrd/`
> (500 files, produced 2026-07-31 13:38–14:08). That production has correctly matched
> GENIE truth and **is** used for the world-volume background campaign.
>
> Verified directly, same event count (656) in both files:
>
> | ntuple | in-tank frac (GENIE vtx) | max\|trueVtx − corrected GENIE vtx\|, in-tank | corr(trueMuonEnergy, trueFSLEnergy), in-tank |
> |---|---|---|---|
> | `bt_v3world.root` (7/30) | 0.8% (5/656) | n/a — only 1 usable event | n/a |
> | `world/fmvmrd/..._world_0.root` (7/31) | 23.5% (154/656) | 0.14 / 0.055 / 0.19 cm | **1.000000**, max\|diff\| 2.05 MeV |
>
> The 0.8% in-tank fraction on the old file is itself the symptom: with GENIE matched to
> the wrong events the vertices are effectively random, so almost nothing lands in the tank.
> On the 7/31 files the WCSim primary reproduces the GENIE vertex to sub-cm, which is only
> possible if the matching is correct.
>
> **Trap that survives in both productions:** `trueVtx*` is the WCSim *primary start point*,
> not the interaction vertex — a sentinel `(0, 14.4602, -168.100)` for ~67% of world events
> and the *tank-entry point* on the r=152.4 cm wall for the rest. Use
> `trueNuIntxVtx_{X,Y,Z}` plus the offset `(0, +14.466, -168.100)` cm to get the interaction
> vertex in tank coordinates. This is implemented as `_augment_origin_columns()` in
> `src/ambe/mc/cc_selection.py`, which also emits `origin_in_tank` and validates the offset
> at runtime.

==============================================================================
WORLD-VOLUME productionv3 -- event-level truth  (656 WCSim events)
==============================================================================
GENIE interactions thrown in world volume : 20000
  -> reached the tank (WCSim events)      : 656  (3.28% yield)
POTs                                      : 5.4902e+16

--- 1. WHERE THE NEUTRINO INTERACTED ---
  Dirt             219   33.38%
  G4_CONCRETE      188   28.66%
  TankWater        149   22.71%
  MRDSteel          89   13.57%
  TankSteel          7    1.07%
  Aluminium          2    0.30%
  Scinti             2    0.30%

  INSIDE tank water :  149   22.71%
  OUTSIDE tank      :  507   77.29%   <-- the world-sample point

--- 2. PARTICLES ENTERING / PRESENT IN THE TANK ---
  gamma         726   39.67%
  neutron       560   30.60%
  proton        254   13.88%
  mu-           189   10.33%
  e-             28    1.53%
  pi+            26    1.42%
  pi0            22    1.20%
  pi-             9    0.49%
  e+              8    0.44%
  mu+             5    0.27%

--- 3. NEUTRONS: how many, and how many from OUTSIDE the tank ---
  total neutrons in/into the tank : 560
  events with >=1                 : 302 / 656  (46.0%)
  from nu intx OUTSIDE the tank   :  414  ( 73.9%)
  from nu intx INSIDE  the tank   :  146  ( 26.1%)

  neutron KE [MeV]           median    mean    <1 MeV   1-10   10-100  100-1000
  OUTSIDE-origin  n= 414      0.54    19.0    58.5%   24.2%   10.4%      7.0%
  INSIDE-origin   n= 146     44.57   103.2     0.7%    8.9%   63.0%     27.4%

--- 4. NEUTRON YIELD BY ORIGIN MATERIAL ---
  material        events  neutrons  n/event  evts>=1n  tank hits
  Dirt               219        83     0.38     29.7%       6584
  G4_CONCRETE        188       130     0.69     45.7%       4647
  TankWater          149       146     0.98     48.3%      14168
  MRDSteel            89       179     2.01     82.0%        325
  TankSteel            7        19     2.71     57.1%        440
  Aluminium            2         3     1.50    100.0%          0
  Scinti               2         0     0.00      0.0%        149

--- 5. TANK LIGHT (from the ntuple, joined by event index) ---
  events with >=1 tank hit : 303 (46.2%)
  total hits 26313 | hits/event mean 40.1 max 268
  CF clusters total        : 694
  nu intx INSIDE tank :  132/ 149 events give light ( 88.6%) | hits  14168 ( 53.8% of all) | clusters  368
  nu intx OUTSIDE tank:  171/ 507 events give light ( 33.7%) | hits  12145 ( 46.2% of all) | clusters  326

--- 6. GENIE-TRUTH MATCHING CHECK (flux nuvtx vs ntuple trueNuIntxVtx) ---
  corr(nuvtxx, trueNuIntxVtx_X) = -0.0197   (n=656)
  corr(nuvtxy, trueNuIntxVtx_Y) = +0.0130   (n=656)
  corr(nuvtxz, trueNuIntxVtx_Z) = -0.0343   (n=656)
  -> ~0 means the ntuple's GENIE branches are matched to the WRONG events.
================================================================================
WORLD-VOLUME productionv3 -- per-hit composition   (25194 hit-slots, 656 events)
================================================================================

--- 1. NEUTRON SIGNAL vs BACKGROUND (all hits) ---
   -5 no-neutron (BACKGROUND)   20759   82.40%
    3 secondary n<-n             2202    8.74%
    1 primary n                  1646    6.53%
    4 secondary n<-other          216    0.86%
    2 secondary n<-p              179    0.71%
    0 dark noise                  192    0.76%

  SIGNAL (neutron lineage, class 1-4) :   4243   16.84%
  BACKGROUND (no neutron, class -5)   :  20759   82.40%
  DARK NOISE                          :    192    0.76%

  [tank-only pilot for reference: SIGNAL 22.76%, BACKGROUND 76.60%, DARK 0.64%]

--- 2. SPLIT BY WHERE THE NEUTRINO INTERACTED ---
                            INSIDE tank water         OUTSIDE tank
  all hits                  13551 ( 53.8%)      11643 ( 46.2%)
  SIGNAL (n lineage)         3002 ( 70.8%)       1241 ( 29.2%)
  BACKGROUND (-5)           10459 ( 50.4%)      10300 ( 49.6%)
  dark noise                   90 ( 46.9%)        102 ( 53.1%)

  INSIDE tank water: 13551 hits -> SIGNAL 22.15% | BACKGROUND 77.18% | DARK  0.66%

  OUTSIDE tank: 11643 hits -> SIGNAL 10.66% | BACKGROUND 88.47% | DARK  0.88%

--- 3. NEUTRON-LINEAGE (SIGNAL) HITS: origin of the neutrino interaction ---
  total signal hits: 4243
    from OUTSIDE the tank :   1241   29.25%   <-- neutrons made outside
    from INSIDE  the tank :   3002   70.75%

  by origin material:
  material        signal hits   bkg hits  signal frac
  TankWater              3002      10459       22.15%
  Dirt                    301       6017        4.74%
  G4_CONCRETE             724       3661       16.35%
  TankSteel                12        410        2.84%
  MRDSteel                179         98       60.47%
  Scinti                   25        114       17.61%

--- 4. NEUTRON CLASS BREAKDOWN, inside vs outside ---
  class                        INSIDE    OUTSIDE
  primary n                      1094        552
  secondary n<-p                  179          0
  secondary n<-n                 1550        652
  secondary n<-other              179         37

--- 5. SIGNAL LINEAGE CHAINS (neutron-lineage hits) ---

  OUTSIDE-origin (n=1241):
     41.26%     512  e- <- gamma <- n
     21.84%     271  e- <- gamma <- n <- n
     21.03%     261  e- <- gamma <- n <- n <- n
      3.95%      49  e- <- gamma <- n <- n <- p
      3.14%      39  e+ <- gamma <- n
      2.42%      30  e- <- gamma <- n <- n <- n <- n
    root ancestor: n 92.0%, p 5.0%, gamma 1.5%, mu- 1.5%

  INSIDE-origin (n=3002):
     33.58%    1008  e- <- gamma <- n
     27.15%     815  e- <- gamma <- n <- n
     11.26%     338  e- <- gamma <- n <- n <- n
      5.03%     151  e- <- gamma <- n <- p
      3.50%     105  e- <- gamma <- n <- pi-
      2.73%      82  e- <- gamma <- n <- n <- pi+
    root ancestor: n 82.7%, p 8.2%, pi- 4.1%, pi+ 3.9%, K- 0.6%, mu- 0.5%

--- 6. BACKGROUND SPECIES (bg_class), inside vs outside ---
  species                INSIDE          OUTSIDE
  muon             6131 (58.62%)     8712 (84.58%)
  photon           2895 (27.68%)     1329 (12.90%)
  chgpion          1089 (10.41%)      235 ( 2.28%)
  proton            244 ( 2.33%)       12 ( 0.12%)
  other              68 ( 0.65%)       12 ( 0.12%)
  kaon               32 ( 0.31%)        0 ( 0.00%)

--- 7. DELAYED WINDOW (t > 10 us): the neutron-capture region ---
  ALL                1978 hits -> SIGNAL 76.95% | BACKGROUND 20.07% | DARK  2.98%
  INSIDE-origin      1299 hits -> SIGNAL 73.83% | BACKGROUND 23.79% | DARK  2.39%
  OUTSIDE-origin      679 hits -> SIGNAL 82.92% | BACKGROUND 12.96% | DARK  4.12%
