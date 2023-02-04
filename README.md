#############################
### Training with Pytorch ###
#############################
# Joshuha Thomas-Wilsker
##############################
# A few notes made while
# writing code to create data
# and train DNN in PyTorch

## Environment settings
Several non-standard libraries must be present in your python environment. To ensure they are present:

*On lxplus* you will need to do this via a virtualenv (example @ http://scikit-hep.org/root_numpy/start.html):

```
python3 -m venv virtualenv3
```
Now you can run:
```
pip install <package>
```

Check whether or not the library dependancies are present on the lxplus machine and if not install them. Typically you will need to install:
```
numba
awkward
uproot
vector
itertools
sklearn
collections
```

There may be more that I've missed.


# Tensors
- Specialised data struct similar to arrays and matrices
- Used for model parameters, inputs and outputs
- Similar to ndarrays but can be run on a GPU or other hardware accelerators
- Optimised for automatic differentiation

## Creating data sets
- Data sets are saved as parquet files
- You can not preserve dtypes with a csv. Limitation of using the .csv format
- Parquet preserve dtypes
- also faster to save/load, less disk space, cross platform support (unlike pythons pickle)

- Custom dataset for files needs to implement:
  __init__
      Runs once when instantiating the object to initialise directory with input, target and transforms
  __len__
      Number f entries in our sample
  __getitem__
      Loads and returns a sample from the dataset. Will only be called when we iterate the object


# Models
- Passing input data to model automatically executes models forward method
