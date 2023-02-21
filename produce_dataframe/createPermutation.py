import ROOT
import time
from array import array
import optparse
import numba as nb
import awkward as ak
import pandas as pd
import vector
import numpy as np
from itertools import combinations, permutations
import uproot
import os

np.finfo(np.dtype("float32"))
skimstore_place = "skim_root/"

class event_truth:
  def __init__(self):

    ##########
    ## RECO ##
    ##########
    self.particle_p4          = []
    self.particle_ids         = []
    self.particle_status      = []
    self.particle_statusflags = []
    self.mother_indices       = []
    self.mother_ids           = []
    self.mother_status        = []
    self.pscalar_lquark_idx   = -1
    self.pscalar_lquark       = vector.obj(pt = -1, eta = -1, phi = -1, mass = -1)
    self.pscalar_bquark_idx   = -1
    self.pscalar_bquark       = vector.obj(pt = -1, eta = -1, phi = -1, mass = -1)
    self.pscalar_lep_idx      = -1
    self.pscalar_lep          = vector.obj(pt = -1, eta = -1, phi = -1, mass = -1)
    #########
    ## GEN ##
    #########
    self.genjet_p4            = []
    self.genjet_hadflav       = []
    self.genjet_partflav      = []
    self.genlep_p4            = []
    self.mindR_bq             = []
    self.mindR_lq             = []
    self.mindR_ll             = []
    self.pscalar_genbjet_idx  = -1
    self.pscalar_genljet_idx  = -1
    self.pscalar_genlep_idx   = -1
    self.pscalar_genbjet      = vector.obj(pt = -1, eta = -1, phi = -1, mass = -1)
    self.pscalar_genljet      = vector.obj(pt = -1, eta = -1, phi = -1, mass = -1)
    self.pscalar_genlep       = vector.obj(pt = -1, eta = -1, phi = -1, mass = -1)
    self.pscalar_genljet_hadflav = -1
    self.pscalar_genbjet_hadflav = -1

  def set_pscalar_quarks(self):
    no_quarks_assigned = False
    for index_ in range(0, len(self.mother_indices)):

      ##############
      ## Daughter ##
      ##############
      daughter_pt          = self.particle_p4[index_].pt
      daughter_eta         = self.particle_p4[index_].eta
      daughter_phi         = self.particle_p4[index_].phi
      daughter_mass        = self.particle_p4[index_].mass
      daughter_id          = abs(self.particle_ids[index_])
      daughter_stat        = self.particle_status[index_]
      daughter_statusflags = self.particle_statusflags[index_]
      isdaughter_HardProcess = (daughter_statusflags>>7)&1
      # Selection1: HardProcess
      if not isdaughter_HardProcess: 
        continue

      ############
      ## Mother ##
      ############
      mother_idx  = self.mother_indices[index_]
      mother_id   = abs(self.particle_ids[mother_idx])
      mother_stat = self.particle_status[mother_idx]
      # Find mother until it has different particle ID with daughters
      while mother_id == daughter_id:
        mother_idx = self.mother_indices[mother_idx]
        mother_id  = abs(self.particle_ids[mother_idx])
      # Selection2: Not an orphan
      if mother_idx < 0:
        continue

      ############
      ## Lepton ##
      ############
      if daughter_id == 11 or daughter_id == 13:
        if mother_id == 24: # W-boson (We need A->qt->qbW)
          W_mother_idx = self.mother_indices[mother_idx]
          W_mother_id  = abs(self.particle_ids[W_mother_idx])
          while W_mother_id == 24:
            W_mother_idx = self.mother_indices[W_mother_idx]
            W_mother_id  = abs(self.particle_ids[W_mother_idx])
          if W_mother_id == 6:
            top_mother_idx = self.mother_indices[W_mother_idx]
            top_mother_id  = abs(self.particle_ids[top_mother_idx])
          while top_mother_id == 6:
            top_mother_idx = self.mother_indices[top_mother_idx]
            top_mother_id  = abs(self.particle_ids[top_mother_idx])
          if top_mother_id == 5000001:
            self.pscalar_lep     = vector.obj(pt=daughter_pt, eta=daughter_eta, phi=daughter_phi, mass=daughter_mass)
            self.pscalar_lep_idx = index_

      ###########
      ## Quark ##
      ###########

      if mother_id == 5000001:
        if daughter_id == 2 or daughter_id == 4:
          self.pscalar_lquark     = vector.obj(pt=daughter_pt, eta=daughter_eta, phi=daughter_phi, mass=daughter_mass)
          self.pscalar_lquark_idx = index_

      elif mother_id == 6:
        top_mother_idx = self.mother_indices[mother_idx]
        top_mother_id  = abs(self.particle_ids[top_mother_idx])
        while top_mother_id == 6:
          top_mother_idx = self.mother_indices[top_mother_idx]
          top_mother_id  = abs(self.particle_ids[top_mother_idx])
          if top_mother_id == 5000001:
            if daughter_id == 5:
              self.pscalar_bquark     = vector.obj(pt=daughter_pt, eta=daughter_eta, phi=daughter_phi, mass=daughter_mass)
              self.pscalar_bquark_idx = index_
  def dr_parton_particle(self):
    j_p4_bq = []
    j_p4_cq = []
    for i in range(0,len(self.genjet_p4)):
      j_p4 = self.genjet_p4[i]
      dR_jlq = j_p4.deltaR(self.pscalar_lquark)
      dR_jbq = j_p4.deltaR(self.pscalar_bquark)
      if dR_jlq < dR_jbq:
        if len(self.mindR_lq)==0:
          self.mindR_lq.append(dR_jlq)
          self.pscalar_genljet         = j_p4
          self.pscalar_genljet_idx     = i
          self.pscalar_genljet_hadflav = self.genjet_hadflav[i]
        elif all(x > dR_jlq for x in self.mindR_lq):
          self.mindR_lq[0] = dR_jlq
          self.pscalar_genljet         = j_p4
          self.pscalar_genljet_idx     = i
          self.pscalar_genljet_hadflav = self.genjet_hadflav[i]
      if dR_jbq < dR_jlq:
        if len(self.mindR_bq)==0:
          self.mindR_bq.append(dR_jbq)
          self.pscalar_genbjet         = j_p4
          self.pscalar_genbjet_idx     = i
          self.pscalar_genbjet_hadflav = self.genjet_hadflav[i]
        elif all(x > dR_jbq for x in self.mindR_bq):
          self.mindR_bq[0] = dR_jbq
          self.pscalar_genljet         = j_p4
          self.pscalar_genljet_idx     = i
          self.pscalar_genljet_hadflav = self.genjet_hadflav[i]
    for lep_ in range(len(self.genlep_p4)):
      l_p4 = self.genlep_p4[lep_]
      dR_ll = l_p4.deltaR(self.pscalar_lep)
      if len(self.mindR_ll)==0:
        self.mindR_ll.append(dR_ll)
        self.pscalar_genlep     = l_p4
        self.pscalar_genlep_idx = lep_
      elif all(x > dR_ll for x in self.mindR_ll):
        self.mindR_ll[0]        = dR_ll
        self.pscalar_genlep     = l_p4
        self.pscalar_genlep_idx = lep_

#@nb.njit
def compute_di_mass(v1, v2):
    return (v1 + v2).mass

def AddPermutation(fin_name, From, To, Tag):


  #################
  ## Data Loader ##
  #################

  # Use uproot to process data
  tree_name = 'Events'
  try:
    events = uproot.open(os.path.join(fin_name) + ":" + tree_name)
  except FileNotFoundError():
    raise FileNotFoundError('Input tree &s not found! ' % (fin_name + ":" + tree_name))

  run_event = To 
  ###############
  ## GenParton ##
  ###############

  GenPart_pt   = events.arrays(['GenPart_pt'],   entry_stop=run_event)['GenPart_pt']
  GenPart_eta  = events.arrays(['GenPart_eta'],  entry_stop=run_event)['GenPart_eta']
  GenPart_phi  = events.arrays(['GenPart_phi'],  entry_stop=run_event)['GenPart_phi']
  GenPart_mass = events.arrays(['GenPart_mass'], entry_stop=run_event)['GenPart_mass']
  GenPart_p4   = vector.zip({'pt':GenPart_pt, 'eta':GenPart_eta, 'phi':GenPart_phi, 'mass':GenPart_mass})
  GenPart_pdgId            = events.arrays(['GenPart_pdgId'], entry_stop=run_event)['GenPart_pdgId']
  GenPart_status           = events.arrays(['GenPart_status'], entry_stop=run_event)['GenPart_status']
  GenPart_statusflags      = events.arrays(['GenPart_statusFlags'], entry_stop=run_event)['GenPart_statusFlags']
  GenPart_genPartIdxMother = events.arrays(['GenPart_genPartIdxMother'], entry_stop=run_event)['GenPart_genPartIdxMother']

  ############
  ## GenJet ##
  ############
  nGenJet     = events.arrays(['nGenJet'],     entry_stop=run_event)['nGenJet']
  GenJet_pt   = events.arrays(['GenJet_pt'],   entry_stop=run_event)['GenJet_pt']
  GenJet_eta  = events.arrays(['GenJet_eta'],  entry_stop=run_event)['GenJet_eta']
  GenJet_phi  = events.arrays(['GenJet_phi'],  entry_stop=run_event)['GenJet_phi']
  GenJet_mass = events.arrays(['GenJet_mass'], entry_stop=run_event)['GenJet_mass']
  GenJet_p4   = vector.zip({'pt':GenJet_pt, 'eta':GenJet_eta, 'phi':GenJet_phi, 'mass':GenJet_mass})
  GenJet_hadflav  = events.arrays(['GenJet_hadronFlavour'], entry_stop=run_event)['GenJet_hadronFlavour']
  GenJet_partflav = events.arrays(['GenJet_partonFlavour'], entry_stop=run_event)['GenJet_partonFlavour']

  ###############
  ## GenLepton ##
  ###############
  nGenLeps    = events.arrays(['nGenDressedLepton'],     entry_stop=run_event)['nGenDressedLepton']
  GenLep_pt   = events.arrays(['GenDressedLepton_pt'],   entry_stop=run_event)['GenDressedLepton_pt']
  GenLep_eta  = events.arrays(['GenDressedLepton_eta'],  entry_stop=run_event)['GenDressedLepton_eta']
  GenLep_phi  = events.arrays(['GenDressedLepton_phi'],  entry_stop=run_event)['GenDressedLepton_phi']
  GenLep_mass = events.arrays(['GenDressedLepton_mass'], entry_stop=run_event)['GenDressedLepton_mass']
  GenLep_p4   = vector.zip({'pt':GenLep_pt, 'eta':GenLep_eta, 'phi':GenLep_phi, 'mass':GenLep_mass})

  #############
  ## RecoJet ##
  #############
  
  nRecoJets      = events.arrays(['nJet'],         entry_stop=run_event)['nJet']
  RecoJet_pt     = events.arrays(['Jet_pt'],       entry_stop=run_event)['Jet_pt']
  RecoJet_eta    = events.arrays(['Jet_eta'],      entry_stop=run_event)['Jet_eta']
  RecoJet_phi    = events.arrays(['Jet_phi'],      entry_stop=run_event)['Jet_phi']
  RecoJet_mass   = events.arrays(['Jet_mass'],     entry_stop=run_event)['Jet_mass']
  RecoJet_p4     = vector.zip({'pt':RecoJet_pt, 'eta':RecoJet_eta, 'phi':RecoJet_phi, 'mass':RecoJet_mass})
  RecoJet_genjetidx = events.arrays(['Jet_genJetIdx'], entry_stop=run_event)['Jet_genJetIdx']
  RecoJet_CvB    = events.arrays(['Jet_btagDeepFlavCvB'], entry_stop=run_event)['Jet_btagDeepFlavCvB']
  RecoJet_CvL    = events.arrays(['Jet_btagDeepFlavCvL'], entry_stop=run_event)['Jet_btagDeepFlavCvL']
  RecoJet_FlavB  = events.arrays(['Jet_btagDeepFlavB'],   entry_stop=run_event)['Jet_btagDeepFlavB']
  TightJet_id    = events.arrays(['tightJets_id_in24'], entry_stop=run_event)['tightJets_id_in24']
  nTightJet      = events.arrays(['n_tight_jet'],       entry_stop=run_event)['n_tight_jet']

  ##################
  ## RecoElectron ##
  ##################

  nElectrons        = events.arrays(['nElectron'],     entry_stop=run_event)['nElectron']
  RecoElectron_pt   = events.arrays(['Electron_pt'],   entry_stop=run_event)['Electron_pt']
  RecoElectron_eta  = events.arrays(['Electron_eta'],  entry_stop=run_event)['Electron_eta']
  RecoElectron_phi  = events.arrays(['Electron_phi'],  entry_stop=run_event)['Electron_phi']
  RecoElectron_mass = events.arrays(['Electron_mass'], entry_stop=run_event)['Electron_mass']
  RecoElectron_p4   = vector.zip({'pt':RecoElectron_pt, 'eta':RecoElectron_eta, 'phi':RecoElectron_phi, 'mass':RecoElectron_mass})

  ##############
  ## RecoMuon ##
  ##############
  nMuons        = events.arrays(['nMuon'],     entry_stop=run_event)['nMuon']
  RecoMuon_pt   = events.arrays(['Muon_pt'],   entry_stop=run_event)['Muon_pt']
  RecoMuon_eta  = events.arrays(['Muon_eta'],  entry_stop=run_event)['Muon_eta']
  RecoMuon_phi  = events.arrays(['Muon_phi'],  entry_stop=run_event)['Muon_phi']
  RecoMuon_mass = events.arrays(['Muon_mass'], entry_stop=run_event)['Muon_mass']
  RecoMuon_p4   = vector.zip({'pt':RecoMuon_pt, 'eta':RecoMuon_eta, 'phi':RecoMuon_phi, 'mass':RecoMuon_mass})

  #############
  ## GenInfo ##
  #############
  df = [] 
  for row in range(From, To):
    print(row)
    ev_truth = event_truth()
    ev_truth.particle_p4          = GenPart_p4[row]
    ev_truth.particle_ids         = GenPart_pdgId[row]
    ev_truth.particle_status      = GenPart_status[row]
    ev_truth.particle_statusflags = GenPart_statusflags[row]
    ev_truth.mother_indices       = GenPart_genPartIdxMother[row]
    ev_truth.genjet_p4            = GenJet_p4[row]
    ev_truth.genjet_hadflav       = GenJet_hadflav[row]
    ev_truth.genjet_partflav      = GenJet_partflav[row]
    ev_truth.genlep_p4            = GenLep_p4[row]

    ev_truth.set_pscalar_quarks()
    if ev_truth.pscalar_lquark_idx == -1 or ev_truth.pscalar_bquark_idx == -1 or ev_truth.pscalar_lep_idx == -1:
      continue
    ev_truth.dr_parton_particle()

    recoj1_idx = -1
    recoj2_idx = -1
    recoJet_p4_dict = {}
    for j_idx in range(0, len(RecoJet_p4[row])):
      if not (j_idx in TightJet_id[row]):
        continue
      recoJet_p4_dict['RecoJet{0}'.format(j_idx)] = RecoJet_p4[row,j_idx]
      if RecoJet_genjetidx[row][j_idx] == ev_truth.pscalar_genbjet_idx:
        scalar_recoj1_p4 = RecoJet_p4[row,j_idx]
        recoj1_idx       = j_idx
      if RecoJet_genjetidx[row][j_idx] == ev_truth.pscalar_genljet_idx:
        scalar_recoj2_p4 = RecoJet_p4[row,j_idx]
        recoj2_idx       = j_idx

    if recoj1_idx == -1 or recoj2_idx == -1:
      continue
    
    reco_leptons_p4_list = []
    for el in RecoElectron_p4[row]:
      reco_leptons_p4_list.append(el)
    for mu in RecoMuon_p4[row]:
      reco_leptons_p4_list.append(mu)
    reco_leptons_p4_list.sort(key=lambda x: x.pt, reverse=True)

    tight_jet_id_skim = TightJet_id[row][:min(nTightJet[row],6)]
    combinations_list = list(permutations(tight_jet_id_skim, 2 ))

    bmatched_jet_pt = []
    bmatched_jet_ptOverMjj = []
    bmatched_jet_eta = []
    bmatched_jet_phi = []
    bmatched_jet_mass = []
    bmatched_jet_massOverMjj = []
    bmatched_jet_CvB = []
    bmatched_jet_CvL = []
    bmatched_jet_FlavB = []
    bmatched_jet_index = []
    lmatched_jet_pt = []
    lmatched_jet_ptOverMjj = []
    lmatched_jet_eta = []
    lmatched_jet_phi = []
    lmatched_jet_mass = []
    lmatched_jet_massOverMjj = []
    lmatched_jet_CvB = []
    lmatched_jet_CvL = []
    lmatched_jet_FlavB = []
    lmatched_jet_index = []
    dR_bmatched_lmatched_jets = []
    dR_bmatched_jet_lep1 = []
    dR_bmatched_jet_lep2 = []
    dR_lmatched_jet_lep1 = []
    dR_lmatched_jet_lep2 = []
    invmass_bjlj = []
    invmass_bjljOverMjj = []
    jet3_pt = []
    jet3_eta = []
    jet3_phi = []
    jet3_mass = []
    jet3_CvL = []
    jet3_CvB = []
    jet3_FlavB = []
    jet3_index = []
    jet4_pt = []
    jet4_eta = []
    jet4_phi = []
    jet4_mass = []
    jet4_CvL = []
    jet4_CvB = []
    jet4_FlavB = []
    jet4_index = []
    leading_lept_pt = []
    leading_lept_eta = []
    leading_lept_phi = []
    leading_lept_mass = []
    subleading_lept_pt = []
    subleading_lept_eta = []
    subleading_lept_phi = []
    subleading_lept_mass = []
    labels = []
    Entry = []

    print(combinations_list)
    for comb_ in combinations_list:
      bmatched_jet_pt.append(RecoJet_pt[row][comb_[0]])
      bmatched_jet_eta.append(RecoJet_eta[row][comb_[0]])
      bmatched_jet_phi.append(RecoJet_phi[row][comb_[0]])
      bmatched_jet_mass.append(RecoJet_mass[row][comb_[0]])
      bmatched_jet_CvB.append(RecoJet_CvB[row][comb_[0]])
      bmatched_jet_CvL.append(RecoJet_CvL[row][comb_[0]])
      bmatched_jet_FlavB.append(RecoJet_FlavB[row][comb_[0]])
      bmatched_jet_index.append(comb_[0])
      lmatched_jet_pt.append(RecoJet_pt[row][comb_[1]])
      lmatched_jet_eta.append(RecoJet_eta[row][comb_[1]])
      lmatched_jet_phi.append(RecoJet_phi[row][comb_[1]])
      lmatched_jet_mass.append(RecoJet_mass[row][comb_[1]])
      lmatched_jet_CvB.append(RecoJet_CvB[row][comb_[1]])
      lmatched_jet_CvL.append(RecoJet_CvL[row][comb_[1]])
      lmatched_jet_FlavB.append(RecoJet_FlavB[row][comb_[1]])
      lmatched_jet_index.append(comb_[1])
      dR_bmatched_lmatched_jets.append(RecoJet_p4[row][comb_[0]].deltaR(RecoJet_p4[row][comb_[1]]))
      dR_bmatched_jet_lep1.append(RecoJet_p4[row][comb_[0]].deltaR(reco_leptons_p4_list[0]))
      dR_bmatched_jet_lep2.append(RecoJet_p4[row][comb_[0]].deltaR(reco_leptons_p4_list[1]))
      dR_lmatched_jet_lep1.append(RecoJet_p4[row][comb_[1]].deltaR(reco_leptons_p4_list[0]))
      dR_lmatched_jet_lep2.append(RecoJet_p4[row][comb_[1]].deltaR(reco_leptons_p4_list[1]))
      j1_p4 = vector.obj(pt=RecoJet_pt[row][comb_[0]], eta=RecoJet_eta[row][comb_[0]], phi=RecoJet_phi[row][comb_[0]], mass=RecoJet_mass[row][comb_[0]])
      j2_p4 = vector.obj(pt=RecoJet_pt[row][comb_[1]], eta=RecoJet_eta[row][comb_[1]], phi=RecoJet_phi[row][comb_[1]], mass=RecoJet_mass[row][comb_[1]])
      invmass_bjlj.append(compute_di_mass(j1_p4, j2_p4))
      
      jet3_index_ = -1
      jet4_index_ = -1
      for idx in tight_jet_id_skim:
        if idx not in [comb_[0], comb_[1]] and jet3_index_ == -1:
          jet3_index_ = idx
          jet3_pt.append(RecoJet_p4[row][idx].pt)
          jet3_eta.append(RecoJet_p4[row][idx].eta)
          jet3_phi.append(RecoJet_p4[row][idx].phi)
          jet3_mass.append(RecoJet_p4[row][idx].mass)
          jet3_CvB.append(RecoJet_CvB[row][idx])
          jet3_CvL.append(RecoJet_CvL[row][idx])
          jet3_FlavB.append(RecoJet_FlavB[row][idx])
        elif idx not in [comb_[0], comb_[1], jet3_index_]:
          jet4_index_ = idx
          jet4_pt.append(RecoJet_p4[row][idx].pt)
          jet4_eta.append(RecoJet_p4[row][idx].eta)
          jet4_phi.append(RecoJet_p4[row][idx].phi)
          jet4_mass.append(RecoJet_p4[row][idx].mass)
          jet4_CvB.append(RecoJet_CvB[row][idx])
          jet4_CvL.append(RecoJet_CvL[row][idx])
          jet4_FlavB.append(RecoJet_FlavB[row][idx])
          break
      if jet4_index_ == -1:
          jet4_pt.append(-9.0)
          jet4_eta.append(-9.0)
          jet4_phi.append(-9.0)
          jet4_mass.append(-9.0)
          jet4_CvB.append(-1.0)
          jet4_CvL.append(-1.0)
          jet4_FlavB.append(RecoJet_FlavB[row][idx])
      jet3_index.append(jet3_index_)
      jet4_index.append(jet4_index_)

      leading_lept_pt.append(reco_leptons_p4_list[0].pt)
      leading_lept_eta.append(reco_leptons_p4_list[0].eta)
      leading_lept_phi.append(reco_leptons_p4_list[0].phi)
      leading_lept_mass.append(reco_leptons_p4_list[0].mass)
      subleading_lept_pt.append(reco_leptons_p4_list[1].pt)
      subleading_lept_eta.append(reco_leptons_p4_list[1].eta)
      subleading_lept_phi.append(reco_leptons_p4_list[1].phi)
      subleading_lept_mass.append(reco_leptons_p4_list[1].mass)

      if (comb_[0] == recoj1_idx) and (comb_[1] == recoj2_idx):
        label = 1
      else:
        label = 0
      labels.append(label)
      Entry.append(row)

     # print(in_array)
     # with torch.no_grad():
     #   print(model(in_array)[0][0].item())
    d_entries = {
       'Entry': Entry,
       'bmatched_jet_index': bmatched_jet_index,
       'lmatched_jet_index': lmatched_jet_index,
       'jet3_index': jet3_index,
       'jet4_index': jet4_index,
       'bmatched_jet_pt': bmatched_jet_pt,
       'bmatched_jet_eta': bmatched_jet_eta,
       'bmatched_jet_phi': bmatched_jet_phi,
       'bmatched_jet_mass': bmatched_jet_mass,
       'bmatched_jet_CvB': bmatched_jet_CvB,
       'bmatched_jet_CvL': bmatched_jet_CvL,
       'bmatched_jet_FlavB': bmatched_jet_FlavB,
       'lmatched_jet_pt': lmatched_jet_pt,
       'lmatched_jet_eta': lmatched_jet_eta,
       'lmatched_jet_phi': lmatched_jet_phi,
       'lmatched_jet_mass': lmatched_jet_mass,
       'lmatched_jet_CvB': lmatched_jet_CvB,
       'lmatched_jet_CvL': lmatched_jet_CvL,
       'lmatched_jet_FlavB': lmatched_jet_FlavB,
       'dR_bmatched_lmatched_jets': dR_bmatched_lmatched_jets,
       'dR_bmatched_jet_lep1': dR_bmatched_jet_lep1,
       'dR_bmatched_jet_lep2': dR_bmatched_jet_lep2,
       'dR_lmatched_jet_lep1': dR_lmatched_jet_lep1,
       'dR_lmatched_jet_lep2': dR_lmatched_jet_lep2,
       'invmass_bjlj': invmass_bjlj,
       'lep1_pt': leading_lept_pt,
       'lep1_eta': leading_lept_eta,
       'lep1_phi': leading_lept_phi,
       'lep1_mass': leading_lept_mass,
       'lep2_pt': subleading_lept_pt,
       'lep2_eta': subleading_lept_eta,
       'lep2_phi': subleading_lept_phi,
       'lep2_mass': subleading_lept_mass,
       'jet3_pt': jet3_pt,
       'jet3_eta': jet3_eta,
       'jet3_phi': jet3_phi,
       'jet3_mass': jet3_mass,
       'jet3_CvB': jet3_CvB,
       'jet3_CvL': jet3_CvL,
       'jet3_FlavB': jet3_FlavB,
       'jet4_pt': jet4_pt,
       'jet4_eta': jet4_eta,
       'jet4_phi': jet4_phi,
       'jet4_mass': jet4_mass,
       'jet4_CvB': jet4_CvB,
       'jet4_CvL': jet4_CvL,
       'jet4_FlavB': jet4_FlavB,
       'label':labels
    }
    df_ = pd.DataFrame(data=d_entries)
    df.append(df_)
  df = pd.concat(df)

  
  print(df)

  df.to_hdf(('dataframe/' + fin_name.split('/')[-1].split('.')[0] + '_' + str(Tag) + '.h5'),'df',mode='w',format='table',data_columns=True)

  return 0
if __name__ == "__main__":
  start = time.time()

  usage = 'usage: %prog [options]'
  parser = optparse.OptionParser(usage)
  parser.add_option('-i','--iin',   dest='iin',   help='input file name', default='ttc_a_200-700GeV_with_geninfo.root', type='string')
  parser.add_option('--from', dest='From', help = 'start point', default = 0, type  = 'int')
  parser.add_option('--to',   dest='To',   help = 'end point', default = 100, type = 'int')
  parser.add_option('--tag',  dest='tag',  help = 'Output tag', default = 0, type = 'int')

  (args,opt) = parser.parse_args()

  os.system('mkdir -p dataframe')

  fileIn = args.iin


  fileIn = os.path.join(skimstore_place, fileIn)

  print ("Input file: ", fileIn)
  AddPermutation(fileIn, args.From, args.To, args.tag)

  end = time.time()
  print( "wall time:", end-start)

