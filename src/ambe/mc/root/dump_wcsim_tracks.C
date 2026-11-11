// Every saved WCSim track of every entry (all triggers) as one flat TTree row ("tracks"),
// plus every tank digit with its absolute time ("digits").
//   root -l -b -q 'dump_wcsim_tracks.C("wcsim.root","tracks.root")'
// Needs libWCSimRoot.so on LD_LIBRARY_PATH and WCSim/include on ROOT_INCLUDE_PATH.
R__LOAD_LIBRARY(libWCSimRoot.so)
#include "WCSimRootEvent.hh"

void dump_wcsim_tracks(const char* in, const char* out)
{
  TFile fin(in);
  TTree* t = (TTree*)fin.Get("wcsimT");
  WCSimRootEvent* ev = new WCSimRootEvent();
  t->SetBranchAddress("wcsimrootevent", &ev);

  TFile fout(out, "RECREATE");
  TTree o("tracks", "saved WCSim tracks");
  Int_t entry, trig, id, parent, pdg, flag, startvol, stopvol, parenttype;
  Float_t t0, t1, e0, e1, x0, y0, z0, x1, y1, z1;
  std::string sproc, eproc;
  o.Branch("entry", &entry);   o.Branch("trig", &trig);     o.Branch("id", &id);
  o.Branch("parent", &parent); o.Branch("pdg", &pdg);       o.Branch("flag", &flag);
  o.Branch("startvol", &startvol); o.Branch("stopvol", &stopvol); o.Branch("parenttype", &parenttype);
  o.Branch("t0", &t0); o.Branch("t1", &t1); o.Branch("e0", &e0); o.Branch("e1", &e1);
  o.Branch("x0", &x0); o.Branch("y0", &y0); o.Branch("z0", &z0);
  o.Branch("x1", &x1); o.Branch("y1", &y1); o.Branch("z1", &z1);
  o.Branch("sproc", &sproc); o.Branch("eproc", &eproc);

  TTree dg("digits", "tank digits: absolute time, earliest photon's direct parent");
  Float_t dt, dq;
  Int_t tube, dparent, noise;
  dg.Branch("entry", &entry); dg.Branch("trig", &trig); dg.Branch("t", &dt); dg.Branch("q", &dq);
  dg.Branch("tube", &tube); dg.Branch("dparent", &dparent); dg.Branch("noise", &noise);

  for (Long64_t i = 0; i < t->GetEntries(); ++i) {
    t->GetEntry(i);
    entry = i;
    TClonesArray* photons = ev->GetTrigger(0)->GetCherenkovHitTimes();
    for (int k = 0; k < ev->GetNumberOfEvents(); ++k) {
      WCSimRootTrigger* tr = ev->GetTrigger(k);
      trig = k;
      double tdate = tr->GetHeader()->GetDate();
      for (int j = 0; j < tr->GetNcherenkovdigihits(); ++j) {
        WCSimRootCherenkovDigiHit* d = (WCSimRootCherenkovDigiHit*)tr->GetCherenkovDigiHits()->At(j);
        if (!d) continue;
        dt = d->GetT() + tdate; dq = d->GetQ(); tube = d->GetTubeId();
        dparent = -9; noise = 0;
        float tfirst = 1e30;
        for (int pid : d->GetPhotonIds()) {
          WCSimRootCherenkovHitTime* ph = (WCSimRootCherenkovHitTime*)photons->At(pid);
          if (ph && ph->GetTruetime() < tfirst) {
            tfirst = ph->GetTruetime(); dparent = ph->GetDirectParentID(); noise = ph->GetIsNoise();
          }
        }
        dg.Fill();
      }
      for (int j = 0; j < tr->GetNtrack(); ++j) {
        WCSimRootTrack* tk = (WCSimRootTrack*)tr->GetTracks()->At(j);
        id = tk->GetId(); parent = tk->GetDirectParentID(); pdg = tk->GetIpnu(); flag = tk->GetFlag();
        startvol = tk->GetStartvol(); stopvol = tk->GetStopvol(); parenttype = tk->GetParenttype();
        t0 = tk->GetTime(); t1 = tk->GetStopTime(); e0 = tk->GetE(); e1 = tk->GetEndE();
        x0 = tk->GetStart(0); y0 = tk->GetStart(1); z0 = tk->GetStart(2);
        x1 = tk->GetStop(0); y1 = tk->GetStop(1); z1 = tk->GetStop(2);
        sproc = tk->GetStartProcess(); eproc = tk->GetEndProcess();
        o.Fill();
      }
    }
  }
  fout.cd();
  o.Write();
  dg.Write();
  fout.Close();
}
