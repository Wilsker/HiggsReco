import os
import sys
import optparse, argparse
import subprocess
import json
import ROOT
from collections import OrderedDict


def GetEntries(pathIn, iin):
  fin = ROOT.TFile(pathIn + "/" + iin, 'R')
  tree_name = 'Events'
  t = fin.Get(tree_name)
  nentries = t.GetEntries()
  return nentries


def CheckStatus(pathIn, iin, tag):
  fin = os.path.join(pathIn, iin.replace(".root", "_" + str(tag) + ".h5"))
  return (not (os.path.getsize(fin) == 0))


def prepare_shell(shell_file, command, condor, FarmDir):

  cwd       = os.getcwd()
  with open('%s/%s'%(FarmDir,shell_file), 'w') as shell:
    shell.write('#!/bin/bash\n')
    shell.write('WORKDIR=%s\n'%cwd)
    shell.write('cd ${WORKDIR}\n')
    shell.write(command)
  condor.write('cfgFile=%s\n'%shell_file)
  condor.write('queue 1\n')

if __name__=='__main__':


  usage = 'usage: %prog [options]'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument("--test", action="store_true")
  args = parser.parse_args()  

  os.system('mkdir -p Farm')
  FarmDir   = 'Farm'
  cwd       = os.getcwd()
  os.system('mkdir -p %s'%FarmDir)
  batchsize = 500

  condor = open('%s/condor.sub'%FarmDir,'w')
  condor.write('output = %s/job_common.out\n'%FarmDir)
  condor.write('error  = %s/job_common.err\n'%FarmDir)
  condor.write('log    = %s/job_common.log\n'%FarmDir)
  condor.write('executable = %s/$(cfgFile)\n'%FarmDir)
  condor.write('requirements = (OpSysAndVer =?= "CentOS7")\n')
  condor.write('request_GPUs = 1\n')
  condor.write('+JobFlavour = "workday"\n')
  condor.write('+MaxRuntime = 7200\n')

  cwd = os.getcwd()

  nEntries = GetEntries('skim_root','ttc_a_200-700GeV_with_geninfo.root')
  for tag in range((nEntries//batchsize)+1):
    command =  "source ${WORKDIR}/env.sh\n"
    command += "python createPermutation.py --iin ttc_a_200-700GeV_with_geninfo.root --from %d --to %d --tag %d\n"%(batchsize*tag, min(batchsize*(tag+1),nEntries), tag)
    shell_file = 'produce_df_%d.sh'%(tag)
    prepare_shell(shell_file, command, condor, FarmDir)

  condor.close()
  if not args.test:
    print ("Submitting Jobs on Condor")
    os.system('condor_submit %s/condor.sub'%FarmDir)

    
