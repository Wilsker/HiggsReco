import ROOT
import time
import os, sys
import math
import json
import optparse
from collections import OrderedDict
from math import sqrt

inputFile_path = "/eos/cms/store/group/phys_top/ExtraYukawa/2018" 

def Slim_module(filein):

  filters = "(nElectron == 1) && (nMuon == 1) && (n_tight_jet > 3)"

  df_filein_tree = ROOT.RDataFrame("Events", os.path.join(inputFile_path, filein))
  df_Out         = df_filein_tree.Filter(str(filters))
  os.system('mkdir -p skim_root/')
  fileOut        = os.path.join('skim_root',filein)
  columns = ROOT.std.vector("string")()
  for c in ('nGenPart','GenPart_pt','GenPart_eta','GenPart_phi','GenPart_mass','GenPart_pdgId','GenPart_status','GenPart_statusFlags','GenPart_genPartIdxMother','nGenJet','GenJet_pt','GenJet_eta','GenJet_phi','GenJet_mass','GenJet_hadronFlavour','GenJet_partonFlavour','nGenDressedLepton','GenDressedLepton_pt','GenDressedLepton_eta','GenDressedLepton_phi','GenDressedLepton_mass','nJet','Jet_pt','Jet_eta','Jet_phi','Jet_mass','Jet_genJetIdx','n_tight_jet','tightJets_id_in24','nElectron','Electron_pt','Electron_eta','Electron_phi','Electron_mass','nMuon','Muon_pt','Muon_eta','Muon_phi','Muon_mass','Jet_btagDeepFlavCvB','Jet_btagDeepFlavCvL','Jet_btagDeepFlavB'):
    columns.push_back(c)
  df_Out.Snapshot("Events",fileOut,columns)

if __name__ == "__main__":
  start = time.time()
  start1 = time.clock()

  usage = 'usage: %prog [options]'
  parser = optparse.OptionParser(usage)
  parser.add_option('-i','--iin',   dest='iin',   help='input file name', default='ttc_a_200-700GeV_with_geninfo.root', type='string')
  (args,opt) = parser.parse_args()


  iin = args.iin

  Slim_module(iin)


  end = time.time()
  end1 = time.clock()
  print( "wall time:", end-start)
  print( "process time:", end1-start1)
