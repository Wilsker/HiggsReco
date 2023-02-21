import torch
import ROOT
import time
from array import array
import optparse
import numba as nb
import awkward as ak
import pandas as pd
import vector
import numpy as np
from torch import nn
from itertools import combinations, permutations
import uproot
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import h5py
import os


skimstore_place = "skim_root/"
h5py_place      = "dataframe/"
output_place    = "MergeOutput/"

class NeuralNetwork(nn.Module):
    def __init__(self,nvars):
        # return a temporary object of superclass so we can call superclass' methods
        super(NeuralNetwork, self).__init__()
        # Initialise layers
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(), #Flattens contiguous range of dimensions into a tensor
            nn.Linear(nvars,24),
            nn.ReLU(),
            nn.Linear(24,12),
            nn.ReLU(),
            nn.Linear(12,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,1),
        )
    # Method to implement operations on input data
    # Passing input data to model automatically executes models forward method
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def AddEntry(fin_name):


  # Load DNN 
  model = NeuralNetwork(30)
  model.load_state_dict(torch.load('data/saved_model.pt'))
  model.eval()

  df = []
  # Scan target file
  FileList = os.listdir(h5py_place)
  for f in FileList:
    if (fin_name in f and 'h5' in f):
      print(f)
      df_ = pd.read_hdf(os.path.join(h5py_place,f))
      df.append(df_)
  # Prepare dataframe

  df = pd.concat(df)
  print(df)

  print(df.keys())
  # Data Preprocessing
  colnames = list(df.keys())
  ct = ColumnTransformer(
  [('StandardScaler', StandardScaler(), colnames[5:-1] )],
      remainder='drop'# Drop nontransformed columns
  )
  index_ = df[colnames[:5]]
  label_ = df[colnames[-1]]
  result_ = ct.fit_transform(df)
  result_ = np.c_[index_, result_, label_]
  transformed_df = pd.DataFrame(result_,columns=colnames)
  transformed_df = transformed_df.astype({'Entry':"int",'bmatched_jet_index':'int','lmatched_jet_index':'int','jet3_index':'int','jet4_index':'int','label':'int'})
  print(transformed_df)
  transformed_df.to_hdf(os.path.join(output_place,fin_name + '.h5'),'df',mode='w',format='table',data_columns=True)
  input_columns_ = [
    'bmatched_jet_pt','bmatched_jet_eta','bmatched_jet_phi','bmatched_jet_mass',
    'lmatched_jet_pt','lmatched_jet_eta','lmatched_jet_phi','lmatched_jet_mass',
    'dR_bmatched_lmatched_jets','dR_bmatched_jet_lep1','dR_bmatched_jet_lep2','dR_lmatched_jet_lep1','dR_lmatched_jet_lep2',
    'invmass_bjlj',
    'lep1_pt','lep1_eta','lep1_phi','lep1_mass',
    'lep2_pt','lep2_eta','lep2_phi','lep2_mass',
    'jet3_pt','jet3_eta','jet3_phi','jet3_mass',
    'jet4_pt','jet4_eta','jet4_phi','jet4_mass'
  ]
  source_combs = transformed_df[input_columns_].values
  X_input      = torch.tensor(source_combs, dtype=torch.float32)
  predicted    = torch.sigmoid(model(X_input)).detach().numpy()
  predicted    = np.array(predicted.reshape(len(predicted)))
  transformed_df.insert(1, "predicted", predicted)
  print(transformed_df)
  idx = transformed_df.groupby(['Entry'])['predicted'].transform(max) == transformed_df['predicted']
  df = transformed_df[idx]


  df = df.sort_values(by=['Entry'])
  df = df.set_index("Entry")  
  print(df)
  print("accuracy: %.4f"%(df['label'].sum()/len(df)))


  return 0
if __name__ == "__main__":
  start = time.time()

  usage = 'usage: %prog [options]'
  parser = optparse.OptionParser(usage)
  parser.add_option('-i','--iin',   dest='iin',   help='input file name', default='ttc_a_200-700GeV_with_geninfo', type='string')
  (args,opt) = parser.parse_args()

  os.system("mkdir -p %s"%output_place)
  fileIn = args.iin

  print ("Input file: ", fileIn)
  AddEntry(fileIn)

  end = time.time()
  print( "wall time:", end-start)

