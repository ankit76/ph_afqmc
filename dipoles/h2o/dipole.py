import sys
import os
import numpy as np
import csv
import pandas as pd
import h5py

np.set_printoptions(precision=7, linewidth=1000, suppress=True)

# read integrals
norbs = 0
nobs = 0
ints = [ ]
#constants = np.zeros(3)
intFilename = str(sys.argv[1])
with h5py.File(intFilename, 'r') as fh5:
  header = fh5['header'][:]
  norbs = header[0]
  nobs = header[1]
  print(f'norbs: {norbs},  nobs: {nobs}')
  constants = fh5['constants'][:]
  #print(f'constants: {constants}')
  for i in range(nobs):
    ints.append(fh5[f'ints_{i}'][:])

#for i in range(nobs):
#  print(f'{i}:\n {ints[i]}\n')
#exit(0)

# read dice rdm and calculate properties
rdm_dice = np.zeros((norbs, norbs))
with open('spatial1RDM.0.0.txt') as fh:
  next(fh)
  for line in fh:
    ls = line.split()
    rdm_dice[int(ls[0]), int(ls[1])] = float(ls[2])
    rdm_dice[int(ls[1]), int(ls[0])] = float(ls[2])

obs_dice = constants.copy()
for n in range(nobs):
  obs_dice[n] += np.trace(np.dot(rdm_dice, ints[n]))

print(f'variational obs_dice: {obs_dice}')
#exit(0)

# read afqmc rdm and calculate observables
fcount = 0
observables_afqmc = [ ]
weights = [ ]
for filename in os.listdir():
  if (filename.startswith('rdm_')):
    fcount += 1
    with open(filename) as fh:
      weights.append(float(fh.readline()))
    cols = list(range(norbs))
    df = pd.read_csv(filename, delim_whitespace=True, usecols=cols, header=None, skiprows=1)
    rdm_i = df.to_numpy()
    #print(f'{filename}:\n{rdm_i}\n')
    obs_i = constants.copy()
    for n in range(nobs):
      obs_i[n] += np.trace(np.dot(rdm_i, ints[n]))
    #print(f'{filename}: {obs_i}')
    observables_afqmc.append(obs_i)

#print(f'file count: {fcount}')
weights = np.array(weights)
observables_afqmc = np.array(observables_afqmc)
#print(observables)
obsMean = np.zeros(nobs)
obsError = np.zeros(nobs)
v1 = weights.sum()
v2 = (weights**2).sum()
for n in range(nobs):
  obsMean[n] = np.multiply(weights, observables_afqmc[:, n]).sum() / v1
  obsError[n] = (np.multiply(weights, (observables_afqmc[:, n] - obsMean[n])**2).sum() / (v1 - v2 / v1) / (fcount - 1))**0.5
  #obsMean[n] = np.average(observables_afqmc[:, n])
  #obsErr[n] = np.std(observables_afqmc[:, n]) / np.sqrt(fcount - 1)
print(f'mixed obs_afqmc: {obsMean}')
print(f'extrapolated obs: {2*obsMean - obs_dice}')
print(f'errors: {obsError}')
