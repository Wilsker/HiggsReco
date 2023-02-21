## Step1. Slim ntuple
First, you need to slim your root file. Results are stored in `skim_root`
```
python slim.py
```
## Step2. Produce permutation
Source `env.sh` for the environment. `createPermutation.py` handle this job and use `runCondor.py` to submit it to condor. Results are stored in `dataframe`
```
python runCondor.py
```
## Step3. Merge Result
Source `env.sh` for the environment. Results are stored in `MergeOutput`
```
python MergeResult.py 
```
