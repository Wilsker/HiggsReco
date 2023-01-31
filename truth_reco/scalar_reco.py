import argparse, json, math, os, pickle, uuid, time, subprocess
# Using numbas just-in-time python compiler
import numba as nb
import awkward as ak
import numpy as np
import pandas as pd
import uproot
import vector
import itertools
import matplotlib
import sklearn
matplotlib.use('Agg') # Fix for $> _tkinter.TclError: couldn't connect to display "localhost:36.0"
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from itertools import combinations, permutations
from collections import OrderedDict, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

class event_truth:
    """ class for events truth information """
    def __init__(self):
        # Add instances of per event parton level info here
        self.particle_p4 = []
        self.particle_ids = []
        self.particle_status = []
        self.particle_statusflags = []
        self.mother_indices = []
        self.mother_ids = []
        self.mother_status = []
        self.pscalar_lquark_idx = -1
        self.pscalar_lquark = vector.obj(pt=-1, eta=-1, phi=-1, mass=-1)
        self.pscalar_bquark_idx = -1
        self.pscalar_bquark = vector.obj(pt=-1, eta=-1, phi=-1, mass=-1)
        self.pscalar_lquark_lep = -1
        self.pscalar_lep = vector.obj(pt=-1, eta=-1, phi=-1, mass=-1)
        # Add instances of per event particle level info here
        self.genjet_p4 = []
        self.genjet_hadflav = []
        self.genjet_partflav = []
        self.genlep_p4 = []
        self.mindR_bq = []
        self.mindR_lq = []
        self.mindR_ll = []
        self.pscalar_genbjet_idx = -1
        self.pscalar_genljet_idx = -1
        self.pscalar_genlep_idx = -1
        self.pscalar_genbjet = vector.obj(pt=-1, eta=-1, phi=-1, mass=-1)
        self.pscalar_genljet = vector.obj(pt=-1, eta=-1, phi=-1, mass=-1)
        self.pscalar_genlep = vector.obj(pt=-1, eta=-1, phi=-1, mass=-1)
        self.pscalar_genljet_hadflav = -1
        self.pscalar_genbjet_hadflav = -1

    # method assigning b and c/u quarks to a0 decay using truth information
    def set_pscalar_quarks(self):
        no_quarks_assigned = False
        for index_ in range(0,len(self.mother_indices)):
            # set events daughter particle info
            daughter_pt = self.particle_p4[index_].pt
            daughter_eta = self.particle_p4[index_].eta
            daughter_phi = self.particle_p4[index_].phi
            daughter_mass = self.particle_p4[index_].mass
            daughter_id = abs(self.particle_ids[index_])
            daughter_stat = self.particle_status[index_]
            daughter_statusflags = self.particle_statusflags[index_]

            # Get boolean to see if a chosen gen status flag is on or off
            isdaughter_HardProcess = (daughter_statusflags>>7)&1
            # decoded_flag_boolean = ( status_flag_value >> genstatusflag_bit )&1
            # number of right-shifts i.e. genstatusflag_bit, found here:
            # https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/genparticles_cff.py#L64

            # If you want to check/print, ensure leading zeros kept in binary (visualistion puproses only)
            #original_statusflags = format(daughter_statusflags,'015b')
            #shifted_statusflags = format((daughter_statusflags>>7),'015b')

            # Only look at particles from hard process
            if not isdaughter_HardProcess:
                continue

            # Set particles mother info
            mother_idx = self.mother_indices[index_]
            mother_id = abs(self.particle_ids[mother_idx])
            mother_stat = self.particle_status[mother_idx]

            # If a particles mother has the same pdgid iterate up the chain until new particle is found
            while mother_id == daughter_id:
                mother_idx = self.mother_indices[mother_idx]
                mother_id = abs(self.particle_ids[mother_idx])

            # If the mother is -ve, particle is an orphan
            if mother_idx<0:
                continue

            # If lepton
            if daughter_id == 11 or daughter_id == 13:
                # Check ancestory and keep if it has come from a0->top->bW leg
                if mother_id == 24:
                    # Check origin of W
                    W_mother_idx = self.mother_indices[mother_idx]
                    W_mother_id = abs(self.particle_ids[W_mother_idx])
                    while W_mother_id == 24:
                        W_mother_idx = self.mother_indices[W_mother_idx]
                        W_mother_id = abs(self.particle_ids[W_mother_idx])
                    if W_mother_id == 6:
                        # Check origin of top
                        top_mother_idx = self.mother_indices[W_mother_idx]
                        top_mother_id = abs(self.particle_ids[top_mother_idx])
                        # If tops mother is also top, iterate until next particle is found
                        while top_mother_id == 6:
                            top_mother_idx = self.mother_indices[top_mother_idx]
                            top_mother_id = abs(self.particle_ids[top_mother_idx])
                        # If top originated from A/H, save top daughter as true decendent of A/H
                        if top_mother_id == 5000001:
                            self.pscalar_lep = vector.obj(pt=daughter_pt, eta=daughter_eta, phi=daughter_phi, mass=daughter_mass)
                            self.pscalar_lep_idx = index_

            # If mother is pseudoscalar, save daughters pdgid and status
            if mother_id == 5000001:
                # We only want the up or charm quarks from the hardprocess a0 decay
                if daughter_id == 2 or daughter_id == 4:
                    self.pscalar_lquark = vector.obj(pt=daughter_pt, eta=daughter_eta, phi=daughter_phi, mass=daughter_mass)
                    self.pscalar_lquark_idx = index_

            elif mother_id == 6:
                # Check origin of top
                top_mother_idx = self.mother_indices[mother_idx]
                top_mother_id = abs(self.particle_ids[top_mother_idx])
                # If tops mother is also top, iterate until next particle is found
                while top_mother_id == 6:
                    top_mother_idx = self.mother_indices[top_mother_idx]
                    top_mother_id = abs(self.particle_ids[top_mother_idx])
                # If top originated from A/H, save top daughter as true decendent of A/H
                if top_mother_id == 5000001:
                    if daughter_id == 5:
                        self.pscalar_bquark = vector.obj(pt=daughter_pt, eta=daughter_eta, phi=daughter_phi, mass=daughter_mass)
                        self.pscalar_bquark_idx = index_

    # method assigning a0 assigned decay products to particle jets
    def dr_parton_particle(self):
        j_p4_bq = []
        j_p4_cq = []
        # Loop over all gen jets
        for i in range(0,len(self.genjet_p4)):

            # For each gen jet
            j_p4 = self.genjet_p4[i]

            # Calculate dR with light/b quark assigned to a0 decay
            dR_jlq = j_p4.deltaR(self.pscalar_lquark)
            dR_jbq = j_p4.deltaR(self.pscalar_bquark)

            # If closest to light (c/u) quark
            if dR_jlq<dR_jbq:
                # Check if all other dR values previously appended are larger
                if len(self.mindR_lq)==0:
                    self.mindR_lq.append(dR_jlq)
                    self.pscalar_genljet = j_p4
                    self.pscalar_genljet_idx = i
                    self.pscalar_genljet_hadflav = self.genjet_hadflav[i]
                elif all( x>dR_jlq for x in self.mindR_lq):
                    # if so replace
                    self.mindR_lq[0] = dR_jlq
                    self.pscalar_genljet = j_p4
                    self.pscalar_genljet_idx = i
                    self.pscalar_genljet_hadflav = self.genjet_hadflav[i]
            # If closest to b quark
            if dR_jbq<dR_jlq:
                if len(self.mindR_bq)==0:
                    self.mindR_bq.append(dR_jbq)
                    self.pscalar_genbjet = j_p4
                    self.pscalar_genbjet_idx = i
                    self.pscalar_genbjet_hadflav = self.genjet_hadflav[i]
                elif all( x>dR_jbq for x in self.mindR_bq):
                    self.mindR_bq[0] = dR_jbq
                    self.pscalar_genbjet = j_p4
                    self.pscalar_genbjet_idx = i
                    self.pscalar_genbjet_hadflav = self.genjet_hadflav[i]

        for lep_ in range(len(self.genlep_p4)):
            # for each gen lepton
            l_p4 = self.genlep_p4[lep_]
            # calculate dR with lepton assigned to a0 decay
            dR_ll = l_p4.deltaR(self.pscalar_lep)
            # if first or smallest in dR, replace current value
            if len(self.mindR_ll)==0:
                self.mindR_ll.append(dR_ll)
                self.pscalar_genlep = l_p4
                self.pscalar_genlep_idx = lep_
            elif all(x>dR_ll for x in self.mindR_ll):
                self.mindR_ll[0] = dR_ll
                self.pscalar_genlep = l_p4
                self.pscalar_genlep_idx = lep_

# Makes simple histograms from arrays
def make_hist(array, name, nbins, min_bin, max_bin, x_units):
    f, ax = plt.subplots()
    ax.hist(array, nbins, range=(min_bin,max_bin), density=False, edgecolor='black',color='red')
    #ax.tick_params(axis='x', which='minor')
    # axis ticks
    ax.xaxis.set_major_locator(MultipleLocator(100))
    f.suptitle(name)
    ax.set(ylabel='# entries')
    ax.set(xlabel=x_units)
    savename = name+x_units
    savename = savename.replace("-","")
    savename = savename.replace("(","")
    savename = savename.replace(")","")
    savename = savename.replace("/","")
    savename = savename.replace(" ","")
    plt.savefig(savename)
    plt.clf()

def make_comparison_hist(array1, array2, name, nbins, min_bin, max_bin, x_units):
    f, ax = plt.subplots()
    plot_title = name.replace('_',' ')
    ax.hist(array1,nbins,range=(min_bin,max_bin),edgecolor='black',color='red',alpha=0.5,label='correct(S)',density=True)
    ax.hist(array2,nbins,range=(min_bin,max_bin),edgecolor='black',color='blue',alpha=0.5,label='incorrect(B)',density=True)
    #ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set(ylabel='# entries')
    ax.set(xlabel=x_units)
    ax.legend()
    f.suptitle(plot_title)
    plt.savefig('SvsB_'+name+'.png')
    plt.clf()

def get_min_max_bins(arr1_, arr2_):
    min_list = [arr1_.min(),arr2_.min()]
    max_list = [arr1_.max(),arr2_.max()]
    min_bin = min(min_list)
    max_bin = max(max_list)
    return min_bin, max_bin

# For histograms with pre-computed bins
def make_hist_pre(array, name, nbins, labels, x_units):
    f, ax = plt.subplots()
    x_vals = [x for x in range(0,nbins)]
    ax.bar(x_vals, array)
    ax.tick_params(axis='x', which='minor')
    # axis ticks
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels)
    f.suptitle(name)
    ax.set(ylabel='# weighted entries')
    ax.set(xlabel=x_units)
    name.strip("(")
    name.strip(")")
    name.strip("/")
    plt.savefig(name.replace(" ",""))
    plt.clf()

# Use numba decorator njit to ensure time consuming functions
# are run in 'no python' mode. Functions are compiled to machine
# code just-in-time for execution, while the rest of the code is
# run at native machine speed.
@nb.njit
def combine_two_vectors(v1, v2):
    v3 = v1 + v2
    return v3
@nb.njit
def combine_three_vectors(v1, v2, v3):
    v3 = v1 + v2 + v3
    return v3
@nb.njit
def compute_di_mass(v1, v2):
    return (v1 + v2).mass
@nb.njit
def compute_tri_mass(v1, v2, v3):
    return (v1 + v2 + v3).mass

def load_data(input_path, outdir, output_format, num_events, preprocess_data):
    # list files
    filenames = [
    #'ttc_a_1000_rtu04_highmass.root'
    'ttc_a_200-700GeV_with_geninfo.root'
    ]

    for files in filenames:
        # Input file full path
        input_full_path = os.path.join(input_path, files)
        ttree_name = 'Events'

        try:
            events = uproot.open(input_full_path+':'+ttree_name)
        except FileNotFoundError():
            raise FileNotFoundError('Input tree %s not found!' % (input_full_path+':'+tree_name))

        # Use to limit the number of entries run over
        test_entries = num_events

        # Basic selection
        selection_string = "(nElectron == 1) & (nMuon == 1) & (nJet > 3)"

        # Create GenPart collections
        # Speed up code by removing any non-essential variables from here
        GenPart_pt = events.arrays(['GenPart_pt'], selection_string, entry_stop=test_entries)['GenPart_pt']
        GenPart_eta = events.arrays(['GenPart_eta'], selection_string, entry_stop=test_entries)['GenPart_eta']
        GenPart_phi = events.arrays(['GenPart_phi'], selection_string, entry_stop=test_entries)['GenPart_phi']
        GenPart_mass = events.arrays(['GenPart_mass'], selection_string, entry_stop=test_entries)['GenPart_mass']
        GenPart_p4 = vector.zip({'pt':GenPart_pt, 'eta':GenPart_eta, 'phi':GenPart_phi, 'mass':GenPart_mass})

        GenPart_pdgId = events.arrays(['GenPart_pdgId'], selection_string, entry_stop=test_entries)['GenPart_pdgId']
        GenPart_status = events.arrays(['GenPart_status'], selection_string, entry_stop=test_entries)['GenPart_status']
        GenPart_statusflags = events.arrays(['GenPart_statusFlags'], selection_string, entry_stop=test_entries)['GenPart_statusFlags']
        GenPart_genPartIdxMother = events.arrays(['GenPart_genPartIdxMother'], selection_string, entry_stop=test_entries)['GenPart_genPartIdxMother']

        # Create GenJet 4-vectors
        nGenJet = events.arrays(['nGenJet'], selection_string, entry_stop=test_entries)['nGenJet']
        GenJet_pt = events.arrays(['GenJet_pt'], selection_string, entry_stop=test_entries)['GenJet_pt']
        GenJet_eta = events.arrays(['GenJet_eta'], selection_string, entry_stop=test_entries)['GenJet_eta']
        GenJet_phi = events.arrays(['GenJet_phi'], selection_string, entry_stop=test_entries)['GenJet_phi']
        GenJet_mass = events.arrays(['GenJet_mass'], selection_string, entry_stop=test_entries)['GenJet_mass']
        GenJet_p4 = vector.zip({'pt': GenJet_pt, 'eta': GenJet_eta, 'phi': GenJet_phi, 'mass': GenJet_mass})
        GenJet_hadflav = events.arrays(['GenJet_hadronFlavour'], selection_string, entry_stop=test_entries)['GenJet_hadronFlavour']
        GenJet_partflav = events.arrays(['GenJet_partonFlavour'], selection_string, entry_stop=test_entries)['GenJet_partonFlavour']

        # Create GenDressedLepton 4-vectors
        nGenLeps = events.arrays(['nGenDressedLepton'], selection_string, entry_stop=test_entries)['nGenDressedLepton']
        GenLep_pt = events.arrays(['GenDressedLepton_pt'], selection_string, entry_stop=test_entries)['GenDressedLepton_pt']
        GenLep_eta = events.arrays(['GenDressedLepton_eta'], selection_string, entry_stop=test_entries)['GenDressedLepton_eta']
        GenLep_phi = events.arrays(['GenDressedLepton_phi'], selection_string, entry_stop=test_entries)['GenDressedLepton_phi']
        GenLep_mass = events.arrays(['GenDressedLepton_mass'], selection_string, entry_stop=test_entries)['GenDressedLepton_mass']
        GenLep_p4 = vector.zip({'pt':GenLep_pt, 'eta':GenLep_eta, 'phi':GenLep_phi, 'mass':GenLep_mass})

        # Create reco jet 4-vectors
        nRecoJets = events.arrays(['nJet'], selection_string, entry_stop=test_entries)['nJet']
        RecoJet_pt = events.arrays(['Jet_pt'], selection_string, entry_stop=test_entries)['Jet_pt']
        RecoJet_eta = events.arrays(['Jet_eta'], selection_string, entry_stop=test_entries)['Jet_eta']
        RecoJet_phi = events.arrays(['Jet_phi'], selection_string, entry_stop=test_entries)['Jet_phi']
        RecoJet_mass = events.arrays(['Jet_mass'], selection_string, entry_stop=test_entries)['Jet_mass']
        RecoJet_p4 = vector.zip({'pt':RecoJet_pt, 'eta':RecoJet_eta, 'phi':RecoJet_phi, 'mass':RecoJet_mass})
        RecoJet_genjetidx = events.arrays(['Jet_genJetIdx'], selection_string, entry_stop=test_entries)['Jet_genJetIdx']

        # Create reco electron 4-vectors
        nElectrons = events.arrays(['nElectron'], selection_string, entry_stop=test_entries)['nElectron']
        RecoElectron_pt = events.arrays(['Electron_pt'], selection_string, entry_stop=test_entries)['Electron_pt']
        RecoElectron_eta = events.arrays(['Electron_eta'], selection_string, entry_stop=test_entries)['Electron_eta']
        RecoElectron_phi = events.arrays(['Electron_phi'], selection_string, entry_stop=test_entries)['Electron_phi']
        RecoElectron_mass = events.arrays(['Electron_mass'], selection_string, entry_stop=test_entries)['Electron_mass']
        RecoElectron_p4 = vector.zip({'pt':RecoElectron_pt, 'eta':RecoElectron_eta, 'phi':RecoElectron_phi, 'mass':RecoElectron_mass})

        # Create reco muon 4-vectors
        nMuons = events.arrays(['nMuon'], selection_string, entry_stop=test_entries)['nMuon']
        RecoMuon_pt = events.arrays(['Muon_pt'], selection_string, entry_stop=test_entries)['Muon_pt']
        RecoMuon_eta = events.arrays(['Muon_eta'], selection_string, entry_stop=test_entries)['Muon_eta']
        RecoMuon_phi = events.arrays(['Muon_phi'], selection_string, entry_stop=test_entries)['Muon_phi']
        RecoMuon_mass = events.arrays(['Muon_mass'], selection_string, entry_stop=test_entries)['Muon_mass']
        RecoMuon_p4 = vector.zip({'pt':RecoMuon_pt, 'eta':RecoMuon_eta, 'phi':RecoMuon_phi, 'mass':RecoMuon_mass})

        # Code producing some statistics/plots on the jet multiplicities
        #make_hist(nRecoJets, 'nRecoJets', 20, 0, 20, '# reco jets')
        #make_hist(nGenJet, 'nGenJet', 20, 0, 20, '# gen jets')
        #reco_dict = defaultdict(int)
        #for njets in nRecoJets:
            #reco_dict['njets{0}'.format(njets)] += 1
        #print('dict: \n', reco_dict)

        # Dataframe to be used in training
        training_df = []
        candidate_genjet_inv_mass = []
        candidate_reco_inv_mass = []
        candidate_reco_jet_indices = []
        recoj1_index = []
        recoj2_index = []
        sum_cand_reco = []
        cand_jets_lead2=0
        cand_jets_lead3=0
        cand_jets_lead5=0
        cand_jets_lead7=0
        cand_jets_lead9=0

        # Make the candidate at reco level using gen-reco matching
        for row in range(0,len(RecoJet_genjetidx)):
            # Set truth info for event
            ev_truth = event_truth()
            ev_truth.particle_p4 = GenPart_p4[row]
            ev_truth.particle_ids = GenPart_pdgId[row]
            ev_truth.particle_status = GenPart_status[row]
            ev_truth.particle_statusflags = GenPart_statusflags[row]
            ev_truth.mother_indices = GenPart_genPartIdxMother[row]
            ev_truth.genjet_p4 = GenJet_p4[row]
            ev_truth.genjet_hadflav = GenJet_hadflav[row]
            ev_truth.genjet_partflav = GenJet_partflav[row]
            ev_truth.genlep_p4 = GenLep_p4[row]

            # Assign quarks to particle jets
            ev_truth.set_pscalar_quarks()
            if ev_truth.pscalar_lquark_idx == -1 or ev_truth.pscalar_bquark_idx == -1 or ev_truth.pscalar_lep_idx == -1:
                #print('problem with scalar associated lep/quark assignement at gen level')
                continue

            # If any of the expected fermions were not found, skip event
            # May occur if status label requirements are not met
            #if ev_truth.pscalar_lquark.pt == -1 or ev_truth.pscalar_bquark.pt == -1 or ev_truth.pscalar_lep.pt == -1:
            #    continue

            # The following function must come after the parton level information is checked
            ev_truth.dr_parton_particle()

            recoJet_p4_dict = {}
            for j_idx in range(0,len(RecoJet_p4[row])):
                # Create reco jet dict
                recoJet_p4_dict['RecoJet{0}'.format(j_idx)] = RecoJet_p4[row,j_idx]

                # If reco jets genjet partners index, matches index assigned to candidate jets
                # take reco jet as part of reco level candidate
                if RecoJet_genjetidx[row][j_idx] == ev_truth.pscalar_genbjet_idx:
                    scalar_recoj1_p4 = RecoJet_p4[row,j_idx]
                    recoj1_idx = j_idx
                if RecoJet_genjetidx[row][j_idx] == ev_truth.pscalar_genljet_idx:
                    scalar_recoj2_p4 = RecoJet_p4[row,j_idx]
                    recoj2_idx = j_idx

            candidate_reco_jet_indices.append([recoj1_idx,recoj2_idx])

            # Check how often candidate jets fall into leading X jet bin
            sum_cand_reco.append(recoj1_idx+recoj2_idx)
            if recoj1_idx+recoj2_idx <= 1:
                cand_jets_lead2+=1
            if recoj1_idx+recoj2_idx <= 3:
                cand_jets_lead3+=1
            if recoj1_idx+recoj2_idx <= 5:
                cand_jets_lead5+=1
            if recoj1_idx+recoj2_idx <= 7:
                cand_jets_lead7+=1
            if recoj1_idx+recoj2_idx <= 9:
                cand_jets_lead9+=1

            # min dR matching of gen level candidate lepton and reco level leptons
            reco_leptons_p4_list = []

            # collect all reco-level leptons
            for el in RecoElectron_p4[row]:
                reco_leptons_p4_list.append(el)
            for mu in RecoMuon_p4[row]:
                reco_leptons_p4_list.append(mu)

            # Get 4-vector of reco lepton dR matched to GenLep
            scalar_recolep_p4 = min(reco_leptons_p4_list, key=lambda x: ev_truth.pscalar_genlep.deltaR(x))

            # convert to 'vector' class (needs to be a class known to numba to run in decorated function)
            recoj0_ = vector.obj(pt=scalar_recoj1_p4.pt, eta=scalar_recoj1_p4.eta, phi=scalar_recoj1_p4.phi, mass=scalar_recoj1_p4.mass)
            recoj1_ = vector.obj(pt=scalar_recoj2_p4.pt, eta=scalar_recoj2_p4.eta, phi=scalar_recoj2_p4.phi, mass=scalar_recoj2_p4.mass)
            recolep_ = vector.obj(pt=scalar_recolep_p4.pt, eta=scalar_recolep_p4.eta, phi=scalar_recolep_p4.phi, mass=scalar_recolep_p4.mass)
            reco_dijet_lep_p4_mass = compute_tri_mass(recoj0_, recoj1_, recolep_)
            candidate_reco_inv_mass.append(reco_dijet_lep_p4_mass)

            # GenJet + GenDressedLepton invariant mass
            genlj_ = vector.obj(pt=ev_truth.pscalar_genljet.pt, eta=ev_truth.pscalar_genljet.eta, phi=ev_truth.pscalar_genljet.phi, mass=ev_truth.pscalar_genljet.mass)
            gebbj_ = vector.obj(pt=ev_truth.pscalar_genbjet.pt, eta=ev_truth.pscalar_genbjet.eta, phi=ev_truth.pscalar_genbjet.phi, mass=ev_truth.pscalar_genbjet.mass)
            genlep_ = vector.obj(pt=ev_truth.pscalar_genlep.pt, eta=ev_truth.pscalar_genlep.eta, phi=ev_truth.pscalar_genlep.phi, mass=ev_truth.pscalar_genlep.mass)
            genjet_dijet_lep_p4_mass = compute_tri_mass(genlj_, gebbj_, genlep_)
            candidate_genjet_inv_mass.append(genjet_dijet_lep_p4_mass)
            recoj1_index.append(recoj1_idx)
            recoj2_index.append(recoj2_idx)

            # Now we start creating output dataset
            # Create dict of various combinations of all n reco jets in event = n(n+1)/2
            combinations_list = list(combinations(recoJet_p4_dict.items(), 2 ))

            # Sort list of reco leptons according to pt for input into dataframe
            reco_leptons_p4_list.sort(key=lambda x: x.pt, reverse=True)

            # If combinations reco jet indices match H/A candidate reco jets
            # assign the combination signal, else it's a background combination
            bmatched_jet_pt = []
            bmatched_jet_eta = []
            bmatched_jet_phi = []
            bmatched_jet_mass = []
            lmatched_jet_pt = []
            lmatched_jet_eta = []
            lmatched_jet_phi = []
            lmatched_jet_mass = []
            dR_bmatched_lmatched_jets = []
            dR_bmatched_jet_lep1 = []
            dR_bmatched_jet_lep2 = []
            dR_lmatched_jet_lep1 = []
            dR_lmatched_jet_lep2 = []
            invmass_bjlj = []
            jet3_pt = []
            jet3_eta = []
            jet3_phi = []
            jet3_mass = []
            jet4_pt = []
            jet4_eta = []
            jet4_phi = []
            jet4_mass = []
            leading_lept_pt = []
            leading_lept_eta = []
            leading_lept_phi = []
            leading_lept_mass = []
            subleading_lept_pt = []
            subleading_lept_eta = []
            subleading_lept_phi = []
            subleading_lept_mass = []
            labels = []

            for comb_ in combinations_list:
                # Combinations of form comb_[which jet][key @dimension='0' or 4-vector @dimension='1']
                n_candidates = 0
                # Check if the index of the jets @ positions 0 or 1 are the same as either candidate reco index
                # [7:] strips 'RecoJet' from key
                if int(comb_[0][0][7:]) in [recoj1_idx,recoj2_idx]:
                    n_candidates+=1
                if int(comb_[1][0][7:]) in [recoj1_idx,recoj2_idx]:
                    n_candidates+=1
                # Signal = any combination where jets in position 1 and 2 had the same index as the candidate reco jets
                if n_candidates == 2:
                    label = 1
                else:
                    label = 0

                bmatched_jet_pt.append(comb_[0][1].pt)
                bmatched_jet_eta.append(comb_[0][1].eta)
                bmatched_jet_phi.append(comb_[0][1].phi)
                bmatched_jet_mass.append(comb_[0][1].mass)

                lmatched_jet_pt.append(comb_[1][1].pt)
                lmatched_jet_eta.append(comb_[1][1].eta)
                lmatched_jet_phi.append(comb_[1][1].phi)
                lmatched_jet_mass.append(comb_[1][1].mass)

                dR_bmatched_lmatched_jets.append( comb_[0][1].deltaR(comb_[1][1]) )

                leading_lept_pt.append(reco_leptons_p4_list[0].pt)
                leading_lept_eta.append(reco_leptons_p4_list[0].eta)
                leading_lept_phi.append(reco_leptons_p4_list[0].phi)
                leading_lept_mass.append(reco_leptons_p4_list[0].mass)

                subleading_lept_pt.append(reco_leptons_p4_list[1].pt)
                subleading_lept_eta.append(reco_leptons_p4_list[1].eta)
                subleading_lept_phi.append(reco_leptons_p4_list[1].phi)
                subleading_lept_mass.append(reco_leptons_p4_list[1].mass)

                dR_bmatched_jet_lep1.append( comb_[0][1].deltaR(reco_leptons_p4_list[0]) )
                dR_bmatched_jet_lep2.append( comb_[0][1].deltaR(reco_leptons_p4_list[1]) )
                dR_lmatched_jet_lep1.append( comb_[1][1].deltaR(reco_leptons_p4_list[0]) )
                dR_lmatched_jet_lep2.append( comb_[1][1].deltaR(reco_leptons_p4_list[1]) )

                tempj0_ = vector.obj(pt=comb_[0][1].pt, eta=comb_[0][1].eta, phi=comb_[0][1].phi, mass=comb_[0][1].mass)
                tempj1_ = vector.obj(pt=comb_[1][1].pt, eta=comb_[1][1].eta, phi=comb_[1][1].phi, mass=comb_[1][1].mass)
                j1j2_combined_mass = compute_di_mass(tempj0_,tempj1_)
                invmass_bjlj.append( j1j2_combined_mass )

                jet3_index = -1
                for key, val in recoJet_p4_dict.items():
                    if int(key[-1]) not in [int(comb_[0][0][-1]),int(comb_[1][0][-1])] and jet3_index == -1:
                        jet3_index = int(key[-1])
                        jet3_pt.append(val.pt)
                        jet3_eta.append(val.eta)
                        jet3_phi.append(val.phi)
                        jet3_mass.append(val.mass)
                    elif int(key[-1]) not in [int(comb_[0][0][-1]),int(comb_[1][0][-1]),jet3_index]:
                        jet4_index = int(key[-1])
                        jet4_pt.append(val.pt)
                        jet4_eta.append(val.eta)
                        jet4_phi.append(val.phi)
                        jet4_mass.append(val.mass)
                        break

                labels.append(label)

            # Dictionary style entries for combination (easy to convert to pandas dataframe)
            d_entries = {
            'Entry': row,
            'bmatched_jet_pt': bmatched_jet_pt,
            'bmatched_jet_eta': bmatched_jet_eta,
            'bmatched_jet_phi': bmatched_jet_phi,
            'bmatched_jet_mass': bmatched_jet_mass,
            'lmatched_jet_pt': lmatched_jet_pt,
            'lmatched_jet_eta': lmatched_jet_eta,
            'lmatched_jet_phi': lmatched_jet_phi,
            'lmatched_jet_mass': lmatched_jet_mass,
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
            'jet4_pt': jet4_pt,
            'jet4_eta': jet4_eta,
            'jet4_phi': jet4_phi,
            'jet4_mass': jet4_mass,
            'label': labels
            }
            df_ = pd.DataFrame(data=d_entries)
            training_df.append(df_)

        make_hist(candidate_reco_inv_mass, 'candidate_reco_jjl_inv_mass', 50, 100, 1100, 'Minv[GeV]')
        make_hist(candidate_genjet_inv_mass, 'candidate_genjet_jjl_inv_mass', 50, 100, 1100, 'Minv[GeV]')
        make_hist(recoj1_index, 'Reco-jet (A/H b genjet matched)', 10, 0, 10, 'Index (pt ordered)')
        make_hist(recoj2_index, 'Reco-jet (A/H light genjet matched)', 10, 0, 10, 'Index (pt ordered)')
        cand_reco_jets_firstNj = [cand_jets_lead2,cand_jets_lead3,cand_jets_lead5,cand_jets_lead7,cand_jets_lead9]
        nj_tick_labels = ['2','3','4','5','6']
        make_hist_pre(cand_reco_jets_firstNj , 'cand_reco_jets_firstNj', len(cand_reco_jets_firstNj), nj_tick_labels, '<=nJ')

        # Concatenate event entries into one list (this may not be necessary and is 'undone' in training code)
        result = pd.concat(training_df)

        #preprocess_data = False

        if preprocess_data == 1:
            # Preprocessing data
            colnames = list(d_entries.keys())
            features_ = result[colnames]
            ct = ColumnTransformer(
            [('StandardScaler', StandardScaler(), colnames[1:-1] )],
                remainder='drop'# Drop nontransformed columns
            )
            index_ = result['Entry']
            result_ = ct.fit_transform(result)
            label_ = result['label']
            result_ = np.c_[index_, result_, label_]
            transformed_df = pd.DataFrame(result_,columns=colnames)
            result = transformed_df


        # Seperate correct and incorrect labels
        df_signal = result.loc[result['label'] == 1]
        df_bkg = result.loc[result['label'] == 0]

        signal_bmatched_jet_pt = df_signal['bmatched_jet_pt']
        bkg_bmatched_jet_pt = df_bkg['bmatched_jet_pt']
        min_b, max_b = get_min_max_bins(signal_bmatched_jet_pt,bkg_bmatched_jet_pt)
        make_comparison_hist(signal_bmatched_jet_pt,bkg_bmatched_jet_pt,'bmatched_jet_pt',20,min_b,max_b,'$p_T[GeV]$')

        signal_lmatched_jet_pt = df_signal['lmatched_jet_pt']
        bkg_lmatched_jet_pt = df_bkg['lmatched_jet_pt']
        min_b, max_b = get_min_max_bins(signal_lmatched_jet_pt,bkg_lmatched_jet_pt)
        make_comparison_hist(signal_lmatched_jet_pt,bkg_lmatched_jet_pt,'lmatched_jet_pt',20,min_b,max_b,'$p_T[GeV]$')

        signal_dR_bmatched_lmatched_jets = df_signal['dR_bmatched_lmatched_jets']
        bkg_dR_bmatched_lmatched_jets = df_bkg['dR_bmatched_lmatched_jets']
        min_b, max_b = get_min_max_bins(signal_dR_bmatched_lmatched_jets,bkg_dR_bmatched_lmatched_jets)
        make_comparison_hist(signal_dR_bmatched_lmatched_jets,bkg_dR_bmatched_lmatched_jets,'dR_bmatched_lmatched_jets',20,min_b,max_b,'dR')

        signal_jet3_pt = df_signal['jet3_pt']
        bkg_jet3_pt = df_bkg['jet3_pt']
        min_b, max_b = get_min_max_bins(signal_jet3_pt,bkg_jet3_pt)
        make_comparison_hist(signal_jet3_pt,bkg_jet3_pt,'jet3_pt',50,min_b,max_b,'$p_T[GeV]$')

        signal_jet4_pt = df_signal['jet4_pt']
        bkg_jet4_pt = df_bkg['jet4_pt']
        min_b, max_b = get_min_max_bins(signal_jet4_pt,bkg_jet4_pt)
        make_comparison_hist(signal_jet4_pt,bkg_jet4_pt,'jet4_pt',50,min_b,max_b,'$p_T[GeV]$')

        signal_bmatched_jet_eta = df_signal['bmatched_jet_eta']
        bkg_bmatched_jet_eta = df_bkg['bmatched_jet_eta']
        min_b, max_b = get_min_max_bins(signal_bmatched_jet_eta,bkg_bmatched_jet_eta)
        make_comparison_hist(signal_bmatched_jet_eta,bkg_bmatched_jet_eta,'bmatched_jet_eta',20,min_b,max_b,f'$\eta$')

        signal_lmatched_jet_eta = df_signal['lmatched_jet_eta']
        bkg_lmatched_jet_eta = df_bkg['lmatched_jet_eta']
        min_b, max_b = get_min_max_bins(signal_lmatched_jet_eta,bkg_lmatched_jet_eta)
        make_comparison_hist(signal_lmatched_jet_eta,bkg_lmatched_jet_eta,'lmatched_jet_eta',20,min_b,max_b,'$\eta$')

        signal_dR_bmatched_jet_lep1 = df_signal['dR_bmatched_jet_lep1']
        bkg_dR_bmatched_jet_lep1 = df_bkg['dR_bmatched_jet_lep1']
        min_b, max_b = get_min_max_bins(signal_dR_bmatched_jet_lep1,bkg_dR_bmatched_jet_lep1)
        make_comparison_hist(signal_dR_bmatched_jet_lep1,bkg_dR_bmatched_jet_lep1,'dR_bmatched_jet_lep1',20,min_b,max_b,'dR')

        signal_dR_bmatched_jet_lep2 = df_signal['dR_bmatched_jet_lep2']
        bkg_dR_bmatched_jet_lep2 = df_bkg['dR_bmatched_jet_lep2']
        min_b, max_b = get_min_max_bins(signal_dR_bmatched_jet_lep2,bkg_dR_bmatched_jet_lep2)
        make_comparison_hist(signal_dR_bmatched_jet_lep2,bkg_dR_bmatched_jet_lep2,'dR_bmatched_jet_lep2',20,min_b,max_b,'dR')

        signal_dR_lmatched_jet_lep1 = df_signal['dR_lmatched_jet_lep1']
        bkg_dR_lmatched_jet_lep1 = df_bkg['dR_lmatched_jet_lep1']
        min_b, max_b = get_min_max_bins(signal_dR_lmatched_jet_lep1,bkg_dR_lmatched_jet_lep1)
        make_comparison_hist(signal_dR_lmatched_jet_lep1,bkg_dR_lmatched_jet_lep1,'dR_lmatched_jet_lep1',20,min_b,max_b,'dR')

        signal_dR_lmatched_jet_lep2 = df_signal['dR_lmatched_jet_lep2']
        bkg_dR_lmatched_jet_lep2 = df_bkg['dR_lmatched_jet_lep2']
        min_b, max_b = get_min_max_bins(signal_dR_lmatched_jet_lep2,bkg_dR_lmatched_jet_lep2)
        make_comparison_hist(signal_dR_lmatched_jet_lep2,bkg_dR_lmatched_jet_lep2,'dR_lmatched_jet_lep2',20,min_b,max_b,'dR')

        signal_invmass_bjlj = df_signal['invmass_bjlj']
        bkg_invmass_bjlj = df_bkg['invmass_bjlj']
        min_b, max_b = get_min_max_bins(signal_invmass_bjlj,bkg_invmass_bjlj)
        make_comparison_hist(signal_invmass_bjlj,bkg_invmass_bjlj,'invmass_bjlj',50,min_b,max_b,'Mass[GeV]')


        # NOTE: You can not preserve dtypes with a csv. Limitation of using csvs.
        # For training try using parquet or hdf5 if you want dtypes preserved
        # Only use .csv if you want a human readable file for inspection
        if output_format=='p':
            # Save as parquet file (fast saving/loading, less disk space, cross platform support)
            result.to_parquet(os.path.join(outdir, 'result_drvar.parquet.gzip'), engine='pyarrow', compression='gzip')
            #transformed_df.to_parquet(os.path.join(outdir, 'result_drvar.parquet.gzip'), engine='pyarrow', compression='gzip')
        if output_format=='c':
            result.to_csv( os.path.join(outdir, 'result.csv') )
            #transformed_df.to_csv( os.path.join(outdir, 'result.csv') )

def main():
    t0 = time.time()
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-i', '--inputs_file_path', dest='inputs_file_path', help='Path to directory containing input files', default='/eos/cms/store/group/phys_top/ExtraYukawa/2018/', type=str)
    parser.add_argument('-o', '--outputs', dest='output_dir', help='Name of output directory', default='', type=str)
    parser.add_argument('-f', '--format', dest='output_format', help= 'Output file format: p = .parquet.gzip, c = .csv', default='p', type=str)
    parser.add_argument('-n', '--nev', dest='num_events', help= 'number of events to run over', default='100000', type=int)
    parser.add_argument('-p', '--preprocess', dest='preprocess_data', default='1', type=int)
    args = parser.parse_args()
    input_file_path_name = args.inputs_file_path
    output_dir = args.output_dir
    output_format = args.output_format
    num_events = args.num_events
    preprocess_data = args.preprocess_data
    print('preprocess_data: ', preprocess_data)

    if os.path.isdir(output_dir) != 1 and len(output_dir)>0:
        os.mkdir(output_dir)

    subprocess.check_output(['pwd', ''])
    subprocess.check_output(['ls', '-l'])
    # Load ttree into .csv including all variables listed in column_headers
    if output_format == 'c':
        print('Creating new data .csv @: %s . . . . ' % (input_file_path_name))
    if output_format == 'p':
        print('Creating new data .parquet @: %s . . . . ' % (input_file_path_name))

    load_data(input_file_path_name, output_dir, output_format, num_events, preprocess_data)

    t1 = time.time()
    print('total run time: %s seconds' % (t1-t0))

if __name__ == '__main__':
    main()
